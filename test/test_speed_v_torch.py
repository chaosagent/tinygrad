import os
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import unittest
import torch
torch.set_num_threads(1)
import time
import numpy as np
np.set_printoptions(linewidth=160)
from functools import partial
from tinygrad.ops import Device
from tinygrad.ops import GlobalCounters
from tinygrad.tensor import Tensor
from tinygrad.nn import Conv2d
from tinygrad.helpers import colored, getenv, CI
from tinygrad.jit import TinyJit
from extra.perf_report import rpt
import pytest

pytestmark = [pytest.mark.exclude_cuda, pytest.mark.exclude_gpu, pytest.mark.exclude_clang]

IN_CHANS = [int(x) for x in getenv("IN_CHANS", "4,16,64").split(",")]

torch_device = torch.device('mps' if getenv("MPS", 0) else ('cuda' if getenv("TORCHCUDA", 0) else 'cpu'))
if str(torch_device) == "mps":
  import torch.mps
  sync = lambda: torch.mps.synchronize()
elif str(torch_device) == "cuda":
  import torch.cuda
  sync = lambda: torch.cuda.synchronize()
else:
  sync = lambda: None

save_ops, save_mem = 0, 0
CNT = 20
def helper_test_speed(f1, *args):
  global save_ops, save_mem
  ets = []
  ret = None
  cache_defeat = np.zeros((2048,2048))
  for i in range(CNT):
    del ret

    # operation cache defeats
    args = [(x+1).realize() if isinstance(x, Tensor) else (None if x is None else (x+1)) for x in args]

    # force syncing
    [x.numpy() if isinstance(x, Tensor) or str(torch_device) == "cpu" else x.cpu().numpy() for x in args if x is not None]

    # clear 32MB global memory cache (CPU and global memory only)
    cache_defeat += 1

    # manual pre sync
    if isinstance(args[0], Tensor): Device[args[0].device].synchronize()
    else: sync()

    GlobalCounters.global_ops = 0
    GlobalCounters.global_mem = 0
    st = time.perf_counter()
    ret = f1(*args)
    if isinstance(ret, Tensor): Device[ret.device].synchronize()
    else: sync()
    et = (time.perf_counter() - st) * 1000
    if i >= 1: ets.append(et)
    if GlobalCounters.global_ops:
      save_ops, save_mem = GlobalCounters.global_ops, GlobalCounters.global_mem
  return ret.numpy() if isinstance(ret, Tensor) else ret.cpu().numpy(), np.min(ets)

def helper_test_generic_square(name, N, f1, f2, onearg=False):
  torch.manual_seed(0)
  torch_a = (torch.rand(N, N) - 0.5).to(torch_device)
  torch_b = (torch.rand(N, N) - 0.5).to(torch_device) if not onearg else None

  tiny_a = Tensor(torch_a.cpu().numpy())
  tiny_b = Tensor(torch_b.cpu().numpy()) if not onearg else None

  helper_test_generic(f"{name:30s} {N:5d}x{N:5d}", f1, (torch_a, torch_b), TinyJit(lambda a,b:f2(a,b).realize()), (tiny_a, tiny_b))

def helper_test_matvec(name, N, M):
  torch.manual_seed(0)
  dt = torch.float32
  torch_a = (torch.rand(N, dtype=dt) - 0.5).to(torch_device)
  torch_b = (torch.rand(N, M, dtype=dt) - 0.5).to(torch_device)

  tiny_a = Tensor(torch_a.cpu().numpy())
  tiny_b = Tensor(torch_b.cpu().numpy())

  helper_test_generic(f"{name:30s} {N:5d}x{M:5d}", lambda a,b: a@b, (torch_a, torch_b), TinyJit(lambda a,b:(a@b).realize()), (tiny_a, tiny_b))

prefix = None
def helper_test_generic(name, f1, f1_args, f2, f2_args):
  global prefix
  with torch.no_grad():
    val_torch, et_torch = helper_test_speed(f1, *f1_args)
  val_tinygrad, et_tinygrad = helper_test_speed(f2, *f2_args)

  flops = save_ops*1e-6
  mem = save_mem*1e-6
  np.testing.assert_allclose(val_tinygrad, val_torch, atol=1e-3, rtol=1e-3)
  rpt.log_perf(name, et_tinygrad, flops, mem, baseline=et_torch)  # don't log performance results if test failed

def helper_test_conv(bs, in_chans, out_chans, kernel_size, img_size_y, img_size_x, padding=0):
  torch.manual_seed(0)
  torch_dat = torch.rand(bs, in_chans, img_size_y, img_size_x).to(torch_device)
  torch_conv = torch.nn.Conv2d(in_chans, out_chans, kernel_size, bias=None, padding=padding).to(torch_device)

  tiny_dat = Tensor(torch_dat.cpu().numpy())
  tiny_conv = Conv2d(in_chans, out_chans, kernel_size, bias=None, padding=padding)
  tiny_conv.weight = Tensor(torch_conv.weight.detach().cpu().numpy())

  def f1(torch_dat): return torch_conv(torch_dat)
  def f2(tiny_dat): return tiny_conv(tiny_dat).realize()
  helper_test_generic(f"conv bs:{bs:3d} chans:{in_chans:3d} -> {out_chans:3d} k:{kernel_size} img:{img_size_y}x{img_size_x}", f1, (torch_dat,), TinyJit(f2), (tiny_dat,))

@unittest.skipIf(getenv("BIG") == 0, "no big tests")
class TestBigSpeed(unittest.TestCase):
  def test_add(self):
    def f(a, b): return a+b
    helper_test_generic_square('add', 8192, f, f)
  def test_exp(self):
    def f(a, b): return a.exp()
    helper_test_generic_square('exp', 8192, f, f, onearg=True)
  def test_gemm_2048(self):
    def f(a, b): return a @ b
    helper_test_generic_square('gemm', 2048, f, f)
  def test_gemm_4096(self):
    def f(a, b): return a @ b
    helper_test_generic_square('gemm', 4096, f, f)
  def test_large_conv_1x1(self): helper_test_conv(bs=32, in_chans=128, out_chans=128, kernel_size=1, img_size_y=128, img_size_x=128)
  def test_large_conv_3x3(self): helper_test_conv(bs=4, in_chans=128, out_chans=128, kernel_size=3, img_size_y=128, img_size_x=128, padding=1)
  def test_large_conv_3x3_matrix(self):
    for image_size in [8, 16, 32, 64, 128]:
      for bs in range(1, 5):
        for in_chans in range(1, 11):
          for out_chans in range(1, 11):
            helper_test_conv(bs=bs, in_chans=2**in_chans, out_chans=2**out_chans, kernel_size=3, img_size_y=image_size, img_size_x=image_size, padding=1)
  def test_large_conv_3x3_matrix_wino(self):
    for bs in [1, 4, 128, 512]:
      for in_chans in [1, 4, 5, 6, 7, 8, 9, 10]:
        for image_size in [8, 32, 128, 224]:
          out_chans=in_chans
          helper_test_conv(bs=bs, in_chans=2**in_chans, out_chans=2**out_chans, kernel_size=3, img_size_y=image_size, img_size_x=image_size, padding=1)
  def test_large_conv_5x5(self): helper_test_conv(bs=4, in_chans=128, out_chans=128, kernel_size=5, img_size_y=132, img_size_x=132)
  def test_matvec_4096_16384(self): helper_test_matvec('matvec_4096_16384', 4096, 16384)
  def test_matvec_16384_4096(self): helper_test_matvec('matvec_16384_4096', 16384, 4096)

@unittest.skipIf(getenv("BIG") == 1, "only big tests")
class TestSpeed(unittest.TestCase):
  def test_sub(self):
    def f(a, b): return a-b
    helper_test_generic_square('sub', 4096, f, f)

  @unittest.skipIf(getenv("CI","")!="" and Device.DEFAULT == "WEBGPU", "breaking on webgpu CI")
  def test_pow(self):
    def f(a, b): return a.pow(b)
    helper_test_generic_square('pow', 2048, f, f)

  def test_sum(self):
    def f(a, b): return a.sum()
    helper_test_generic_square('sum', 2048, f, f, onearg=True)
    helper_test_generic_square('sum', 4096, f, f, onearg=True)

  def test_partial_sum(self):
    R = 256
    def f(a, b): return a.reshape(int(4096//R), int(4096*R)).sum(axis=1)
    helper_test_generic_square('partial_sum', 4096, f, f, onearg=True)

  @unittest.skip("not really used in models")
  def test_cumsum(self):
    def f0(a, b): return a.cumsum(axis=0)
    def f1(a, b): return a.cumsum(axis=1)
    helper_test_generic_square('cumsum_0', 256, f0, f0, onearg=True)
    helper_test_generic_square('cumsum_1', 256, f1, f1, onearg=True)

  def test_cat(self):
    helper_test_generic_square('cat_0', 256, lambda x,y: torch.cat((x,y),dim=0), lambda x,y: x.cat(y,dim=0))
    helper_test_generic_square('cat_1', 256, lambda x,y: torch.cat((x,y),dim=1), lambda x,y: x.cat(y,dim=1))

  def test_array_packing(self):
    N = 2048
    def f(a, b): return a.reshape(N, N // 32, 32).permute(1,0,2).contiguous()
    helper_test_generic_square('array_packing', N, f, f, onearg=True)

  def test_permute(self):
    for N in [1024, 4096]:
      # this is a 64MB tensor, M1 L1 cache is 128kB
      # to fit easily in L1, rotations should be 128x128 chunks. 128x128 is also the AMX size
      def f(a, b): return a.permute(1,0).contiguous()
      helper_test_generic_square('permute', N, f, f, onearg=True)

  def test_double_permute(self):
    N = 64
    torch.manual_seed(0)
    torch_a = (torch.rand(N, N, N, N) - 0.5).to(torch_device)
    tiny_a = Tensor(torch_a.cpu().numpy())
    def f(a): return a.permute(1,0,3,2).contiguous()
    helper_test_generic(f"double_permute {tiny_a.shape}", f, (torch_a,), TinyJit(lambda a: f(a).realize()), (tiny_a,))

  def test_neg(self):
    def f(a, b): return -a
    helper_test_generic_square('neg', 4096, f, f, onearg=True)

  def test_exp(self):
    def f(a, b): return a.exp()
    helper_test_generic_square('exp', 2048, f, f, onearg=True)

  def test_relu(self):
    def f(a, b): return a.relu()
    helper_test_generic_square('relu', 4096, f, f, onearg=True)

  def test_max(self):
    def f(a, b): return a.max()
    helper_test_generic_square('max', 4096, f, f, onearg=True)

  def test_mul_sum(self):
    def f(a, b): return (a*b).sum()
    helper_test_generic_square('mul_sum', 4096, f, f)

  def test_add(self):
    for N in [1, 1024, 4096]:
      def f(a, b): return a + b
      helper_test_generic_square('add', N, f, f)

  def test_add_constant(self):
    def f(a, b): return a+2.0
    helper_test_generic_square('add_constant', 4096, f, f, onearg=True)

  def test_add_sq(self):
    def f(a, b): return a*a + b*b
    helper_test_generic_square('add_sq', 4096, f, f)

  def test_gemm(self):
    def f(a, b): return a @ b
    helper_test_generic_square('gemm', 1024, f, f)

  def test_gemm_small(self):
    def f(a, b): return a @ b
    helper_test_generic_square('gemm', 256, f, f)

  def test_gemm_unrolled(self):
    N = 512
    def f1(a, b): return a@b.T
    def f2(a, b): return (a.reshape(N, 1, N).expand(N, N, N) * b.reshape(1, N, N).expand(N, N, N)).sum(axis=2)
    helper_test_generic_square('gemm_unrolled', N, f1, f2)

  def test_gemm_unrolled_permute_l(self):
    N = 512
    def f1(a, b): return a.T@b.T
    def f2(a, b): return (a.permute(1,0).reshape(N, 1, N).expand(N, N, N) * b.reshape(1, N, N).expand(N, N, N)).sum(axis=2)
    helper_test_generic_square('gemm_unrolled_permute_l', N, f1, f2)

  def test_gemm_unrolled_permute_r(self):
    N = 512
    def f1(a, b): return a@b
    def f2(a, b): return (a.reshape(N, 1, N).expand(N, N, N) * b.permute(1,0).reshape(1, N, N).expand(N, N, N)).sum(axis=2)
    helper_test_generic_square('gemm_unrolled_permute_r', N, f1, f2)

  def test_gemm_unrolled_permute_lr(self):
    N = 512
    def f1(a, b): return a.T@b
    def f2(a, b): return (a.permute(1,0).reshape(N, 1, N).expand(N, N, N) * b.permute(1,0).reshape(1, N, N).expand(N, N, N)).sum(axis=2)
    helper_test_generic_square('gemm_unrolled_permute_lr', N, f1, f2)

  def test_matvec_1024_1024(self): helper_test_matvec('matvec_1024_1024', 1024, 1024)
  def test_matvec_1024_4096(self): helper_test_matvec('matvec_1024_4096', 1024, 4096)
  def test_matvec_4096_1024(self): helper_test_matvec('matvec_4096_1024', 4096, 1024)
  def test_matvec_4096_4096(self): helper_test_matvec('matvec_4096_4096', 4096, 4096)

  def test_openpilot_conv2d(self):
    bs, in_chans, out_chans = 1,12,32
    torch.manual_seed(0)
    torch_dat = torch.rand(bs, 64, 128, 12).to(torch_device)
    torch_conv = torch.nn.Conv2d(in_chans, out_chans, 3, bias=None, padding=1).to(torch_device)

    tiny_dat = Tensor(torch_dat.cpu().numpy())
    tiny_conv = Conv2d(in_chans, out_chans, 3, bias=None, padding=1)
    tiny_conv.weight = Tensor(torch_conv.weight.detach().cpu().numpy())

    def f1(torch_dat): return torch_conv(torch_dat.permute(0,3,1,2))
    def f2(tiny_dat): return tiny_conv(tiny_dat.permute(0,3,1,2)).realize()
    helper_test_generic(f"conv bs:{bs:3d} chans:{in_chans:3d} -> {out_chans:3d} k:3", f1, (torch_dat,), TinyJit(f2), (tiny_dat,))

  def test_conv2d(self):
    for bs in [32]:
      for in_chans in IN_CHANS:
        for out_chans in [32]:
          helper_test_conv(bs, in_chans, out_chans, 3, 34, 34)

if __name__ == '__main__':
  unittest.main()
