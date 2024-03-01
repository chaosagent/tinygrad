from extra import dist
from tinygrad import GlobalCounters, Device, TinyJit
from tinygrad.dtype import dtypes
from tinygrad.helpers import getenv
from tinygrad.tensor import Tensor
from tinygrad.nn.state import get_parameters, get_state_dict
from tinygrad.nn import optim, state
from tqdm import tqdm
import numpy as np
import random
import wandb
import time
import os

BS = getenv("BS", 64)
EVAL_BS = getenv('EVAL_BS', BS)

# fp32 GPUS<=6 7900xtx can fit BS=112

def train_resnet():
  from extra.models.resnet import ResNet50, ResNet18, UnsyncedBatchNorm
  from examples.mlperf.dataloader import batch_load_resnet
  from extra.datasets.imagenet import get_train_files, get_val_files
  from extra.lr_scheduler import PolynomialLR

  seed = getenv('SEED', 42)
  Tensor.manual_seed(seed)

  GPUS = tuple([Device.canonicalize(f'{Device.DEFAULT}:{i}') for i in range(getenv("GPUS", 1))])
  UnsyncedBatchNorm.devices = GPUS
  print(f"Training on {GPUS}")
  for x in GPUS: Device[x]

  # ** model definition **
  num_classes = 1000
  if getenv("SMALL"):
    model, model_name = ResNet18(num_classes), "resnet18"
  else:
    model, model_name = ResNet50(num_classes), "resnet50"

  for v in get_parameters(model):
    v.to_(GPUS)
  parameters = get_parameters(model)

  # ** hyperparameters **
  target = getenv("TARGET", 0.759)
  achieved = False
  epochs = getenv("EPOCHS", 45)
  decay = getenv("DECAY", 2e-4)
  steps_in_train_epoch = (len(get_train_files()) // BS)
  steps_in_val_epoch = (len(get_val_files()) // EVAL_BS)

  # ** Learning rate **
  base_lr = getenv("LR", 8.4 * (BS/2048))

  # ** Optimizer **
  from examples.mlperf.optimizers import LARS
  skip_list = {v for k, v in get_state_dict(model).items() if 'bn' in k or 'bias' in k}
  optimizer = LARS(parameters, base_lr, momentum=.9, weight_decay=decay, skip_list=skip_list)

  # ** LR scheduler **
  lr_warmup_epochs = 5
  scheduler = PolynomialLR(optimizer, base_lr, 1e-4, epochs=epochs * steps_in_train_epoch, warmup=lr_warmup_epochs * steps_in_train_epoch)
  print(f"training with batch size {BS} for {epochs} epochs")

  # ** checkpointing **
  from examples.mlperf.helpers import get_training_state, load_training_state

  start_epoch = 0
  if ckpt:=getenv("RESUME", ""):
    print(f"resuming from {ckpt}")
    load_training_state(model, optimizer, scheduler, state.safe_load(ckpt))
    start_epoch = int(scheduler.epoch_counter.numpy().item() / steps_in_train_epoch)
    print(f"resuming at epoch {start_epoch}")
  elif getenv("TESTEVAL"): model.load_from_pretrained()

  # ** init wandb **
  WANDB = getenv("WANDB")
  if WANDB:
    wandb_config = {
      'BS': BS,
      'EVAL_BS': EVAL_BS,
      'base_lr': base_lr,
      'epochs': epochs,
      'classes': num_classes,
      'decay': decay,
      'train_files': len(get_train_files()),
      'eval_files': len(get_train_files()),
      'steps_in_train_epoch': steps_in_train_epoch,
      'GPUS': GPUS,
      'BEAM': getenv('BEAM'),
      'TEST_TRAIN': getenv('TEST_TRAIN'),
      'TEST_EVAL': getenv('TEST_EVAL'),
      'SYNCBN': getenv('SYNCBN'),
      'model': model_name,
      'optimizer': optimizer.__class__.__name__,
      'scheduler': scheduler.__class__.__name__,
    }
    if getenv("WANDB_RESUME", ""):
      wandb.init(id=getenv("WANDB_RESUME", ""), resume="must", config=wandb_config)
    else:
      wandb.init(config=wandb_config)

  # ** jitted steps **
  input_mean = Tensor([123.68, 116.78, 103.94], device=GPUS, dtype=dtypes.float32).reshape(1, -1, 1, 1)
  # mlperf reference resnet does not divide by input_std for some reason
  # input_std = Tensor([0.229, 0.224, 0.225], device=GPUS, dtype=dtypes.float32).reshape(1, -1, 1, 1)
  def normalize(x): return (x.permute([0, 3, 1, 2]).cast(dtypes.float32) - input_mean).cast(dtypes.default_float)
  @TinyJit
  def train_step(X, Y):
    optimizer.zero_grad()
    X = normalize(X)
    out = model.forward(X)
    loss = out.sparse_categorical_crossentropy(Y, label_smoothing=0.1)
    top_1 = (out.argmax(-1) == Y).sum()
    scheduler.step()
    loss.backward()
    optimizer.step()
    return loss.realize(), out.realize(), top_1.realize()
  @TinyJit
  def eval_step(X, Y):
    X = normalize(X)
    out = model.forward(X)
    loss = out.sparse_categorical_crossentropy(Y, label_smoothing=0.1)
    top_1 = (out.argmax(-1) == Y).sum()
    return loss.realize(), out.realize(), top_1.realize()

  # ** epoch loop **
  for e in range(start_epoch, epochs):
    Tensor.training = True

    it = iter(tqdm(t := batch_load_resnet(batch_size=BS, val=False, shuffle=True, seed=seed*epochs + e), total=steps_in_train_epoch))
    def data_get(it):
      x, y, cookie = next(it)
      # x must realize here, since the shm diskbuffer in dataloader might disappear?
      return x.shard(GPUS, axis=0).realize(), Tensor(y).shard(GPUS, axis=0), cookie

    # ** train loop **
    i, proc = 0, data_get(it)
    st = time.perf_counter()
    while proc is not None:
      if getenv("TESTEVAL"): break

      GlobalCounters.reset()
      proc = (train_step(proc[0], proc[1]), (proc[0], proc[1]), proc[2])

      et = time.perf_counter()
      dt = time.perf_counter()

      try:
        next_proc = data_get(it)
      except StopIteration:
        next_proc = None

      dte = time.perf_counter()

      device_str = proc[0][2].device if isinstance(proc[0][2].device, str) else f"{proc[0][2].device[0]} * {len(proc[0][2].device)}"
      proc, loss_cpu, top_1_acc = None, proc[0][0].numpy(), proc[0][2].numpy().item() / BS  # return cookie
      cl = time.perf_counter()

      tqdm.write(
        f"{i:5} {((cl - st)) * 1000.0:7.2f} ms run, {(et - st) * 1000.0:7.2f} ms python, {(cl - dte) * 1000.0:7.2f} ms {device_str}, {(dte - dt) * 1000.0:6.2f} ms fetch data,"
        f" {loss_cpu:5.2f} loss, {top_1_acc:3.2f} acc, {optimizer.lr.numpy()[0]:.6f} LR, {GlobalCounters.mem_used / 1e9:.2f} GB used, {GlobalCounters.global_ops * 1e-9 / (cl - st):9.2f} GFLOPS")
      if WANDB:
        wandb.log({"lr": optimizer.lr.numpy(),
                   "train/data_time": dte-dt,
                   "train/step_time": cl - st,
                   "train/python_time": et - st,
                   "train/cl_time": cl - dte,
                   "train/loss": loss_cpu,
                   "train/top_1_acc": top_1_acc,
                   "train/GFLOPS": GlobalCounters.global_ops * 1e-9 / (cl - st),
                   "epoch": e + (i + 1) / steps_in_train_epoch,
                   })

      st = cl

      proc, next_proc = next_proc, None

      i += 1

    # ** eval loop **
    if (e+1-getenv("EVAL_START_EPOCH", 2)) % getenv("EVAL_EPOCHS", 4) == 0:
      train_step.reset()  # free the train step memory :(
      eval_loss = []
      eval_times = []
      eval_top_1_acc = []
      Tensor.training = False

      it = iter(tqdm(t := batch_load_resnet(batch_size=EVAL_BS, val=True, shuffle=False), total=steps_in_val_epoch))
      proc = data_get(it)
      while proc is not None:
        GlobalCounters.reset()
        st = time.time()

        proc = (eval_step(proc[0], proc[1]), proc[1], proc[2])

        try:
          next_proc = data_get(it)
        except StopIteration:
          next_proc = None

        eval_loss.append(proc[0][0].numpy().item())
        eval_top_1_acc.append(proc[0][2].numpy().item())
        proc, next_proc = next_proc, None  # drop cookie

        et = time.time()
        eval_times.append(et - st)

      eval_step.reset()
      total_loss = sum(eval_loss) / len(eval_loss)
      total_top_1 = sum(eval_top_1_acc) / len(eval_top_1_acc)
      total_fw_time = sum(eval_times) / len(eval_times)
      tqdm.write(f"eval loss: {total_loss:.2f}, eval time: {total_fw_time:.2f}, eval top 1 acc: {total_top_1:.3f}")
      if WANDB:
        wandb.log({"eval/loss": total_loss,
                  "eval/top_1_acc": total_top_1,
                   "eval/forward_time": total_fw_time,
                   "epoch": e + 1,
        })

      if not achieved and total_top_1 >= target:
        fn = f"./ckpts/{model_name}_cats{num_classes}.safe"
        state.safe_save(state.get_state_dict(model), fn)
        print(f" *** Model saved to {fn} ***")
        achieved = True

      if not getenv("TESTEVAL") and getenv("CKPT"):
        if not os.path.exists("./ckpts"): os.mkdir("./ckpts")
        if WANDB and wandb.run is not None:
          fn = f"./ckpts/{time.strftime('%Y%m%d_%H%M%S')}_{wandb.run.id}_e{e}.safe"
        else:
          fn = f"./ckpts/{time.strftime('%Y%m%d_%H%M%S')}_e{e}.safe"
        print(f"saving ckpt to {fn}")
        state.safe_save(get_training_state(model, optimizer, scheduler), fn)

def train_retinanet():
  # TODO: Retinanet
  pass

def train_unet3d():
  # TODO: Unet3d
  pass

def train_rnnt():
  # TODO: RNN-T
  pass

def train_bert():
  # TODO: BERT
  pass

def train_maskrcnn():
  # TODO: Mask RCNN
  pass

if __name__ == "__main__":
  with Tensor.train():
    for m in getenv("MODEL", "resnet,retinanet,unet3d,rnnt,bert,maskrcnn").split(","):
      nm = f"train_{m}"
      if nm in globals():
        print(f"training {m}")
        globals()[nm]()
