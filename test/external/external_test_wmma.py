import numpy as np
import torch
import unittest

from tinygrad.codegen.kernel import Opt, OptOps
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.helpers import dtypes, getenv
from tinygrad.lazy import vars_from_ast
from tinygrad.ops import Device
from tinygrad.tensor import Tensor

ITER = getenv("ITER", 200)
HALF = getenv("HALF")

def exec_helper(c, t_c, opts):
  s = c.lazydata.schedule()
  c.realize()  # realize an output buffer

  si = s[-1]
  lin = Linearizer(si.ast)
  used_tc = lin.apply_tensor_cores(hand_coded=False)
  assert used_tc
  for opt in opts:
    lin.apply_opt(opt)

  output_buffer = c.lazydata.base.realized
  rawbufs = [output_buffer]+[buf.base.realized for buf in si.inputs]
  lin.linearize()
  var_vals = {k:k.max for k in vars_from_ast(lin.ast)}
  prg = Device[Device.DEFAULT].to_program(lin)
  prg(rawbufs, var_vals)

  np.testing.assert_allclose(t_c, output_buffer.toCPU().reshape(c.shape), atol=1e-2, rtol=1e-2)

def gemm_helper(BS, N, opts):
  a = Tensor.rand(BS, N, N, dtype=dtypes.half if HALF else dtypes.float).realize()
  b = Tensor.rand(BS, N, N, dtype=dtypes.half if HALF else dtypes.float).realize()
  c = a @ b
  t_a, t_b = torch.Tensor(a.numpy()), torch.Tensor(b.numpy())
  t_c = (t_a @ t_b).numpy()
  exec_helper(c, t_c, opts)

def conv_helper(BS, CIN, COUT, N, K, opts):
  x = Tensor.rand(BS, CIN, N, N, dtype=dtypes.half).realize()
  k = Tensor.rand(CIN, COUT, K, K, dtype=dtypes.half).realize()

  c = x.conv2d(k, padding=1)
  t_x, t_k = torch.Tensor(x.numpy()), torch.Tensor(k.numpy())
  t_c = torch.nn.functional.conv2d(t_x, t_k, padding=1)
  exec_helper(c, t_c, opts)

class ExternalTestWmma(unittest.TestCase):
  def test_noopt(self): gemm_helper(1, 1024, [])
  def test_upcast_left(self): gemm_helper(1, 1024, [Opt(OptOps.UPCAST, 0, 4)])
  def test_upcast_right(self): gemm_helper(1, 1024, [Opt(OptOps.UPCAST, 1, 4)])
  def test_upcast_both(self): gemm_helper(1, 1024, [Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UPCAST, 1, 4)])
  def test_upcast_both_flipped(self): gemm_helper(1, 1024, [Opt(OptOps.UPCAST, 1, 4), Opt(OptOps.UPCAST, 0, 4)])
  def test_upcast_bs(self): gemm_helper(4, 1024, [Opt(OptOps.UPCAST, 0, 4)])
  def test_upcast_bs_odd(self): gemm_helper(3, 1024, [Opt(OptOps.UPCAST, 0, 3)])
  def test_upcast_three(self): gemm_helper(4, 1024, [Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.UPCAST, 0, 4)])

class ExternalTestWmmaConv(unittest.TestCase):
  def setUp(self): self.skipTest("WMMA conv not enabled yet")
  def test_noopt(self): conv_helper(4, 64, 64, 224, 3, [])

if __name__ == "__main__":
  unittest.main()

