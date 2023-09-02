import itertools
import time
from tqdm import tqdm
from tinygrad.helpers import getenv, dtypes
from tinygrad.jit import TinyJit
from tinygrad.nn import optim
from tinygrad.nn.state import get_parameters
from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv
from tinygrad.ops import GlobalCounters

def train_resnet():
  # TODO: Resnet50-v1.5
  pass

def train_retinanet():
  # TODO: Retinanet
  pass


from examples.mlperf.metrics import get_dice_score_cpu
def dice_score(pl):
  return get_dice_score_cpu(pl[0], pl[1]).mean()

def train_unet3d(target=0.908, roi_shape=(128, 128, 128)):
  import multiprocessing as mp
  from examples.mlperf.metrics import dice_ce_loss, get_dice_score, one_hot, get_dice_score_cpu
  from extra.datasets.kits19 import (get_train_files, get_val_files, iterate,
                                     sliding_window_inference, preprocess_cache)
  from extra.training import lr_warmup
  from models.unet3d import UNet3D
  import numpy as np

  Tensor.training = True
  dtype = "float16"
  np_dtype = np.float16
  Tensor.default_type = dtypes.half
  in_channels, n_class, BS = 1, 3, 1, # original: 1, 3, 2
  mdl = UNet3D(in_channels, n_class)
  #mdl.load_from_pretrained(dtype=dtype)
  lr_warmup_epochs = 0 # original: 200
  init_lr, lr = 1e-2, 0.8
  max_epochs = 4000
  evaluate_every_epochs = 20
  opt = optim.SGD(get_parameters(mdl), lr=init_lr)

  @TinyJit
  def train_step(image, label):
    opt.zero_grad()
    out = mdl(image).float()
    loss = dice_ce_loss(out, label, n_class)
    loss.backward()
    opt.step()
    return loss.realize(), out.realize()

  for _, image, label, key in iterate(BS=BS, val=False, prewarm=True, roi_shape=roi_shape, epochs=1, dtype=np_dtype):
    preprocess_cache[key] = (np.squeeze(image, axis=0), label, key)
  for _, image, label, key in iterate(BS=BS, val=True, prewarm=True, epochs=1, dtype=np_dtype):
    preprocess_cache[key] = (np.squeeze(image, axis=0), label, key)

  print('cache warmed')

  data_loader = iterate(BS=BS, val=False, roi_shape=roi_shape, epochs=max_epochs, dtype=np_dtype)
  val_data_loader = iterate(BS=BS, val=True, epochs=max_epochs // evaluate_every_epochs, dtype=np_dtype)
  for epoch in range(max_epochs):
    if epoch <= lr_warmup_epochs and lr_warmup_epochs > 0:
      lr_warmup(opt, init_lr, lr, epoch, lr_warmup_epochs)
    if True:
      st = time.monotonic()
      for i, image, label, _ in (t := tqdm(itertools.islice(data_loader, len(get_train_files())//BS), total=len(get_train_files())//BS)):
        GlobalCounters.reset()
        image = Tensor(image)
        label = Tensor(label)
        loss, _ = train_step(image, label)
        et = time.monotonic()
        del image, label
        loss_cpu = loss.numpy().item()
        cl = time.monotonic()
        t.set_description(f"loss {loss_cpu}")
        tqdm.write(f"{i:3d} {(cl - st) * 1000.0:7.2f} ms run, {(et - st) * 1000.0:7.2f} ms python, {(cl - et) * 1000.0:7.2f} ms CL, {loss_cpu:7.2f} loss, {opt.lr.numpy()[0]:.6f} LR, {GlobalCounters.mem_used / 1e9:.2f} GB used, {GlobalCounters.global_ops * 1e-9 / (cl - st):9.2f} GFLOPS")
        st = time.monotonic()

    if (epoch + 1) % evaluate_every_epochs == 0 or False:
      Tensor.training = False
      inference_results = [sliding_window_inference(mdl, image, label) for _, image, label, _ in tqdm(itertools.islice(val_data_loader, len(get_val_files())//BS), total=len(get_val_files())//BS)]
      with mp.Pool(12) as p:
        s = sum(tqdm(p.imap(dice_score, inference_results), total=len(inference_results)))
      val_dice_score = s / len(get_val_files())
      print(f"[Epoch {epoch}] Val dice score: {val_dice_score:.4f}. Target: {target}")
      Tensor.training = True
      if val_dice_score >= target:
        break

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
      globals()[nm]()


