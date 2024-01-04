from extra import dist
from tinygrad.helpers import getenv, dtypes
if __name__ == "__main__":
  if getenv("DIST"):
    dist.preinit()

from extra.dist import collectives
from extra.helpers import cross_process
from tinygrad.tensor import Tensor
from tinygrad.jit import TinyJit
from tinygrad.nn.state import get_parameters, get_state_dict
from tinygrad.nn import optim
from tinygrad.helpers import GlobalCounters, getenv, dtypes
from tqdm import tqdm
import numpy as np
import random
import wandb
import time

FP16 = getenv("FP16", 0)
BS, EVAL_BS, STEPS = getenv("BS", 32), getenv('EVAL_BS', 32), getenv("STEPS", 1000)

def train_resnet():
  # TODO: Resnet50-v1.5
  from models.resnet import ResNet50
  from extra.datasets.imagenet import iterate, get_train_files, get_val_files
  from extra.lr_scheduler import MultiStepLR, LR_Scheduler, Optimizer, PolynomialLR

  def sparse_categorical_crossentropy(out, Y, label_smoothing=0):
    num_classes = out.shape[-1]
    y_counter = Tensor.arange(num_classes, requires_grad=False).unsqueeze(0).expand(Y.numel(), num_classes)
    y = (y_counter == Y.flatten().reshape(-1, 1)).where(-1.0 * num_classes, 0)
    y = y.reshape(*Y.shape, num_classes)
    return (1 - label_smoothing) * out.mul(y).mean() + (-1 * label_smoothing * out.mean())


  def calculate_accuracy(out, Y, top_n):
    out_top_n = np.argpartition(out.cpu().numpy(), -top_n, axis=-1)[:, -top_n:]
    YY = np.expand_dims(Y.numpy(), axis=1)
    YY = np.repeat(YY, top_n, axis=1)

    eq_elements = np.equal(out_top_n, YY)
    top_n_acc = np.count_nonzero(eq_elements) / eq_elements.size * top_n
    return top_n_acc

  seed = getenv('SEED', 42)
  Tensor.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)

  from extra.dist import OOB
  assert OOB is not None or not getenv("DIST"), "OOB should be initialized"
  rank, world_size = getenv("RANK"), getenv("WORLD_SIZE", 1)

  if rank == 0: wandb.init()

  num_classes = 1000
  if FP16:
    Tensor.default_type = dtypes.float16
  model = ResNet50(num_classes)
  if getenv("TESTEVAL"): model.load_from_pretrained()
  if FP16:
    for v in get_parameters(model): v.assign(v.cast(dtypes.float16).realize())
  parameters = get_parameters(model)

  @TinyJit
  def train_step(X, Y):
    optimizer.zero_grad()
    out = model.forward(X)
    loss = sparse_categorical_crossentropy(out, Y, label_smoothing=0.1) * lr_scaler
    loss.backward()

    if getenv("DIST"):
      # sync gradients across ranks
      bucket, offset = [], 0
      for v in parameters:
        if v.grad is not None: bucket.append(v.grad.flatten())
      grads = collectives.allreduce(Tensor.cat(*bucket))
      for v in parameters:
        if v.grad is not None:
          v.grad.assign(grads[offset:offset + v.grad.numel()].reshape(*v.grad.shape))
          offset += v.grad.numel()

    optimizer.step()
    return loss.realize(), out.realize()

  @TinyJit
  def eval_step(X, Y):
    out = model.forward(X)
    loss = sparse_categorical_crossentropy(out, Y, label_smoothing=0.1)
    return loss.realize(), out.realize()

  lr_scaler = 8
  lr_gamma = 0.1
  lr_steps = [30, 60, 80]
  lr_warmup = 5
  base_lr = 0.256 * (BS / 256)  # Linearly scale from BS=256, lr=0.256
  base_lr = 0.1
  epochs = getenv("EPOCHS", 45)
  optimizer = optim.SGD(parameters, base_lr / lr_scaler, momentum=.875, weight_decay=1/2**15)
  steps_in_train_epoch = (len(get_train_files()) // BS) - 1
  steps_in_val_epoch = (len(get_val_files()) // EVAL_BS) - 1
  #scheduler = MultiStepLR(optimizer, [m for m in lr_steps], gamma=lr_gamma, warmup=lr_warmup)
  scheduler = PolynomialLR(optimizer, 0.1, 1e-4, epochs)
  print(f"training with batch size {BS} for {epochs} epochs")
  #for i in range(40):
  #  scheduler.step()
  #  print(scheduler.get_lr().numpy())
  #exit(0)

  for e in range(epochs):
    # train loop
    Tensor.training = True
    scheduler.step()
    print(scheduler.get_lr().numpy()[0])
    dt = time.perf_counter()
    for i, (X, Y, data_time) in enumerate(tqdm(t := cross_process(lambda: iterate(bs=BS, val=False, num_workers=48)), total=steps_in_train_epoch)):
      if getenv("TESTEVAL"): break

      next_Xt, next_Yt = Tensor(X, requires_grad=False), Tensor(Y, requires_grad=False)
      if getenv("DIST"): next_Xt, next_Yt = next_Xt.chunk(world_size, 0)[rank], next_Yt.chunk(world_size, 0)[rank]
      next_Xt, next_Yt = next_Xt.realize(), next_Yt.realize()
      if i == 0: Xt, Yt = next_Xt, next_Yt

      dte = time.perf_counter()

      if i != 0:
        loss_cpu = loss.numpy() / lr_scaler
        cl = time.perf_counter()
        new_st = time.perf_counter()

        if rank == 0:
          tqdm.write(
            f"{i:5} {((cl - st)) * 1000.0:7.2f} ms run, {(et - st) * 1000.0:7.2f} ms python, {(cl - et) * 1000.0:7.2f} ms CL, {(dte - dt) * 1000.0:7.2f} ms fetch data, {loss_cpu:7.2f} loss, {optimizer.lr.numpy()[0] * lr_scaler:.6f} LR, {GlobalCounters.mem_used / 1e9:.2f} GB used, {GlobalCounters.global_ops * 1e-9 / (cl - st):9.2f} GFLOPS")
          wandb.log({"lr": optimizer.lr.numpy() * lr_scaler,
                     "train/data_time": data_time,
                     "train/python_time": et - st,
                     "train/step_time": cl - st,
                     "train/other_time": cl - et,
                     "train/loss": loss_cpu,
                     "train/GFLOPS": GlobalCounters.global_ops * 1e-9 / (cl - st),
                     })
        st = new_st
      else:
        st = time.perf_counter()
      next_X, next_Y = X, Y

      GlobalCounters.reset()
      loss, out = train_step(Xt, Yt)
      Xt, Yt = next_Xt, next_Yt
      et = time.perf_counter()
      dt = time.perf_counter()


    # "eval" loop. Evaluate every 4 epochs, starting with epoch 0
    if e % 1 == 0:
      eval_loss = []
      eval_times = []
      eval_top_1_acc = []
      eval_top_5_acc = []
      Tensor.training = False
      for (X, Y, data_time) in (t := tqdm(cross_process(lambda: iterate(bs=EVAL_BS, val=True, num_workers=48)), total=steps_in_val_epoch)):
        X, Y = Tensor(X, requires_grad=False), Tensor(Y, requires_grad=False)
        if getenv("DIST"):
          X, Y = X.chunk(world_size, 0)[rank], Y.chunk(world_size, 0)[rank]
        st = time.time()
        loss, out = eval_step(X, Y)
        et = time.time()

        top_1_acc = calculate_accuracy(out, Y, 1)
        top_5_acc = calculate_accuracy(out, Y, 5)
        if getenv("DIST"):
          if rank == 0:
            gloss, gtop1, gtop5 = loss, top_1_acc, top_5_acc
            for j in range(1, world_size):
              recv_loss, recv_top1, recv_top5 = OOB.recv(j)
              gloss += recv_loss
              gtop1 += recv_top1
              gtop5 += recv_top5
            loss = gloss / world_size
            top_1_acc = gtop1 / world_size
            top_5_acc = gtop5 / world_size
          elif rank < min(world_size, 5):
            OOB.send((loss, top_1_acc, top_5_acc), 0)
        eval_loss.append(loss.numpy().item())
        eval_times.append(et - st)
        eval_top_1_acc.append(top_1_acc)
        eval_top_5_acc.append(top_5_acc)


      if rank == 0:
        tqdm.write(f"eval loss: {sum(eval_loss) / len(eval_loss):.2f}, eval time: {sum(eval_times) / len(eval_times):.2f}, eval top 1 acc: {sum(eval_top_1_acc) / len(eval_top_1_acc):.2f}, eval top 5 acc: {sum(eval_top_5_acc) / len(eval_top_5_acc):.2f}")
        wandb.log({"eval/loss": sum(eval_loss) / len(eval_loss),
                  "eval/forward_time": sum(eval_times) / len(eval_times),
                  "eval/top_1_acc": sum(eval_top_1_acc) / len(eval_top_1_acc),
                  "eval/top_5_acc": sum(eval_top_5_acc) / len(eval_top_5_acc),
        })

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
  Tensor.training = True

  for m in getenv("MODEL", "resnet,retinanet,unet3d,rnnt,bert,maskrcnn").split(","):
    nm = f"train_{m}"
    if nm in globals():
      print(f"training {m}")
      train_func = globals()[nm]
      break
  else:
    print("please specify MODEL")
    exit(1)

  if not getenv("DIST"):
    train_func()
  else: # distributed
    if getenv("HIP"):
      from tinygrad.runtime.ops_hip import HIP
      devices = [f"hip:{i}" for i in range(HIP.device_count)]
    else:
      from tinygrad.runtime.ops_gpu import CL
      devices = [f"gpu:{i}" for i in range(len(CL.devices))]
    world_size = len(devices)

    # ensure that the batch size is divisible by the number of devices
    assert BS % world_size == 0, f"batch size {BS} is not divisible by world size {world_size}"

    # ensure that the evaluation batch size is divisible by the number of devices
    assert EVAL_BS % world_size == 0, f"evaluation batch size {EVAL_BS} is not divisible by world size {world_size, 5}"

    # init out-of-band communication
    dist.init_oob(world_size)

    # start the processes
    processes = []
    for rank, device in enumerate(devices):
      processes.append(dist.spawn(rank, device, fn=train_func, args=()))
    for p in processes: p.join()

