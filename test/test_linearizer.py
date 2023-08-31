import numpy as np
import unittest

from tinygrad.codegen.linearizer import Linearizer, UOps
from tinygrad.ops import Compiled, Device
from tinygrad.tensor import Tensor
from tinygrad.jit import CacheCollector

class TestLinearizer(unittest.TestCase):
  def test_arg_dedup(self):
    if not isinstance(Device[Device.DEFAULT], Compiled):
      self.skipTest("Only Compiled supports cache")
    a, b = Tensor.randn(4), Tensor.randn(4)
    np_a, np_b = a.numpy(), b.numpy()
    CacheCollector.start()
    c = ((a.shrink(((0, 2),)) - a.shrink(((2, 4),))) - (b.shrink(((0, 2),)) - b.shrink(((2, 4),)))).realize()
    rawbufs = CacheCollector.finish()[0][1]
    assert len(rawbufs) == 3 and set(rawbufs[1:]) == {a.lazydata.realized, b.lazydata.realized}
    np_c = (np_a[:2] - np_a[2:]) - (np_b[:2] - np_b[2:])
    np.testing.assert_allclose(np_c, c.numpy())

  def test_arg_dedup_assign(self):
    if not isinstance(Device[Device.DEFAULT], Compiled):
      self.skipTest("Only Compiled supports cache")
    a, b = Tensor.randn(4), Tensor.randn(4)
    np_a, np_b = a.numpy(), b.numpy()
    CacheCollector.start()
    # the following should add a to b in-place in b.
    b.assign(a + b).realize()
    rawbufs = CacheCollector.finish()[0][1]
    # output buffer must be first rawbuffer
    assert len(rawbufs) == 2 and rawbufs == [b.lazydata.realized, a.lazydata.realized]
    np_c = np_a + np_b
    np.testing.assert_allclose(np_c, b.numpy())

  def test_arg_dedup_assign2(self):
    if not isinstance(Device[Device.DEFAULT], Compiled):
      self.skipTest("Only Compiled supports cache")
    a, b, c = Tensor.randn(4), Tensor.randn(4), Tensor.randn(4)
    np_a, np_b, np_c = a.numpy(), b.numpy(), c.numpy()
    # the following should add a to b in-place in b.
    d = b + c
    b.assign(a + b).realize()
    d = d.realize()
    np_apb = np_a + np_b
    np_d = np_b + np_c
    np.testing.assert_allclose(np_apb, b.numpy())
    np.testing.assert_allclose(np_d, d.numpy())

  def test_load_dedup(self):
    # for different leaves in the AST, the same loads may occur.

    if not isinstance(Device[Device.DEFAULT], Compiled):
      self.skipTest("Only Compiled uses linearizer")

    a = Tensor.randn(4).realize()
    # these are of size 3 to avoid float4 coalesce
    r = a[:-1] + a[1:]
    ast = r.lazydata.op
    r = r.realize()  # realize an output buffer
    k = Linearizer(ast, r.lazydata, Device[Device.DEFAULT].linearizer_opts)
    k.process()
    k.upcast()
    k.linearize()
    num_loads = len([uop for uop in k.uops if uop.uop == UOps.LOAD])
    assert num_loads <= 4, "more load uops than needed"
    assert num_loads >= 4, "unexpected number of uops, maybe this test needs updating?"

  def test_upcast_cse(self):
    # when upcasting, within a subtree, there may be common expressions.

    if not isinstance(Device[Device.DEFAULT], Compiled):
      self.skipTest("Only Compiled uses linearizer")

    a, b = Tensor.randn(1).realize(), Tensor.randn(1).realize()
    r = a.expand([2]) + b.expand([2])
    ast = r.lazydata.op
    r = r.realize()  # realize an output buffer
    k = Linearizer(ast, r.lazydata, Device[Device.DEFAULT].linearizer_opts)
    k.process()
    k.upcast()
    k.linearize()
    num_ops = len([uop for uop in k.uops if uop.uop == UOps.ALU])
    assert num_ops <= 1, "more alu uops than needed"

if __name__ == '__main__':
  unittest.main()
