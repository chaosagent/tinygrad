import unittest
import time
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.helpers import Context

class TestConv(unittest.TestCase):
  def test_simple(self):
    x = Tensor.ones(1,12,128,256).contiguous().realize()
    w = Tensor.ones(32,12,3,3).contiguous().realize()
    ret = x.conv2d(w, stride=(2,2), padding=(1,1)).numpy()
    # it's not 108 around the padding
    assert (ret[:, :, 1:-1, 1:-1] == 108).all()
    assert ret[0,0,0,0] == 48
    assert ret[0,0,0,1] == 72

  def test_simple_rand(self):
    x = Tensor.rand(1,12,128,256)
    w = Tensor.rand(32,12,3,3)
    x.conv2d(w, stride=(2,2), padding=(1,1)).numpy()

  def test_many_simple(self):
    x = Tensor(np.arange(8*2*8).reshape(1,8,2,8).astype(np.float32))
    #w = Tensor(np.arange(8*8*1*1).reshape(8,8,1,1).astype(np.float32))
    w = Tensor.eye(8).reshape((8,8,1,1))
    ret = x.conv2d(w, stride=(1,2), padding=(0,0)).numpy()
    print(ret)

  def test_lazycache(self):
    Tensor.no_grad = True
    x = Tensor.rand(1, 32)
    y = Tensor.rand(32)
    out = x + y.reshape((1,32,1)).reshape((1,32)) + y.reshape((1,32,1)).reshape((1,32))
    out.numpy()
    Tensor.no_grad = False

  def test_simple_biased(self):
    C = 8
    x = Tensor.rand(1,C,5,5)
    w = Tensor.eye(C).reshape((C,C,1,1))
    b = Tensor(np.arange(C).astype(np.float32))
    ret = Tensor.conv2d(x,w,b).relu().conv2d(w,b)

    print(ret.numpy())

  def test_two_binops_no_rerun(self):
    Tensor.no_grad = True
    x = Tensor.randn(1,12,128,256)
    w = Tensor.randn(32,12,3,3)
    out = x.conv2d(w, stride=(2,2), padding=(1,1))
    r1, r2 = out.relu(), (out-1)
    np.testing.assert_allclose(r1.numpy(), np.maximum(out.numpy(), 0))
    np.testing.assert_allclose(r2.numpy(), out.numpy() - 1)
    Tensor.no_grad = False

  def test_two_overlapping_binops_no_rerun(self):
    Tensor.no_grad = True
    x = Tensor.randn(1,12,128,256)
    w = Tensor.randn(32,12,3,3)
    out = x.conv2d(w, stride=(2,2), padding=(1,1))
    r1, r2 = out.relu(), out.elu()
    np.testing.assert_allclose(r1.numpy(), np.maximum(out.numpy(), 0))
    np.testing.assert_allclose(r2.numpy(), np.where(out.numpy() > 0, out.numpy(), (np.exp(out.numpy()) - 1)), atol=1e-5)
    Tensor.no_grad = False

  def test_two_overlapping_binops_no_rerun_wino(self):
    Tensor.no_grad = True
    with Context(WINO=1):
      x = Tensor.randn(1,4,16,16)
      w = Tensor.randn(6,4,3,3)
      out = x.conv2d(w, padding=(1,1))
      r1, r2 = out.relu(), out.elu()
      np.testing.assert_allclose(r1.numpy(), np.maximum(out.numpy(), 0))
      np.testing.assert_allclose(r2.numpy(), np.where(out.numpy() > 0, out.numpy(), (np.exp(out.numpy()) - 1)), atol=1e-5)
    Tensor.no_grad = False

  def test_first_three(self):
    Tensor.no_grad = True
    x = Tensor.rand(1,12,128,256)

    w = Tensor.rand(32,12,3,3)
    x = x.conv2d(w, stride=(2,2), padding=(1,1)).elu()

    w = Tensor.rand(32,1,3,3)
    x = x.conv2d(w, padding=(1,1), groups=32).elu()

    w = Tensor.rand(16,32,1,1)
    x = x.conv2d(w).elu()

    x = x.numpy()
    print(x.shape)
    Tensor.no_grad = False

  def test_elu(self):
    Tensor.no_grad = True
    x = Tensor.rand(1,12,128,256)

    w = Tensor.rand(32,12,3,3)
    x = x.conv2d(w, stride=(2,2), padding=(1,1))

    x = x.elu()

    w = Tensor.rand(32,1,3,3)
    x = x.conv2d(w, padding=(1,1), groups=32)
    x.numpy()
    Tensor.no_grad = False

  def test_reduce_relu(self):
    Tensor.no_grad = True
    x = Tensor.rand(1,12,128,256)
    x = x.sum(keepdim=True).relu()
    x.numpy()
    Tensor.no_grad = False

  def test_bias(self):
    Tensor.no_grad = True
    from tinygrad.nn import Conv2d
    x = Tensor.rand(1,12,128,256)
    c = Conv2d(12, 32, 3)
    x = c(x).relu()
    w = Tensor.uniform(32, 1, 3, 3)
    x = x.conv2d(w, groups=32)
    x.numpy()
    Tensor.no_grad = False

  def test_multiadd(self):
    w = Tensor.rand(32)
    x = Tensor.rand(32).relu()
    (w+x).numpy()

  def test_reorder(self):
    x = Tensor.rand(1,12,128,256)
    w = Tensor.rand(12,12,3,3)
    x = x.conv2d(w, padding=(1,1))
    print(x.shape)
    x = x.reshape((1, 12, 256, 128))
    x += 1
    x += 1
    x = x.reshape((1, 12, 128, 256))
    x.numpy()

  def test_conv_backwards(self):
    from tinygrad import TinyJit, nn, dtypes
    from examples.mlperf.initializers import Conv2dHeNormal
    from tinygrad.helpers import getenv

    cin, cout, k, s, p = 64, 64, 3, 1, 1
    bs, y, x = 128, 56, 56

    #cin, cout, k, s, p = 256, 256, 3, 2, 1
    #bs, y, x = 256, 28, 28

    c1 = Conv2dHeNormal(cin, cout, k, stride=s, padding=p)
    dtypes.default_float = dtypes.half
    c1.weight.requires_grad = True
    #c1.bias.requires_grad = True
    # run
    @TinyJit
    def f():
      img = Tensor.rand(bs, cin, y, x, requires_grad=True, dtype=dtypes.half)
      for i in range(getenv("CNT", 1)):
        c1(img).relu().mean().backward()
        #check_schedule(c1.weight.grad, 3, [c1.weight, c1.bias])
        #check_schedule(img.grad, 3, [c1.weight, c1.bias])
        c1.weight.grad.realize()
        img.grad.contiguous().realize()
      return img.grad
    for i in range(5):
      st = time.perf_counter()
      f().numpy()
      et = time.perf_counter()
      print(f'{et - st}s')

  def test_funny_st(self):
    from tinygrad.shape.shapetracker import ShapeTracker, View
    st = ShapeTracker(views=(View(shape=(256, 1, 256, 256, 3, 14, 2, 3, 14, 2), strides=(50176, 0, 1, 0, 0, 3584, 0, 0, 256, 0), offset=0, mask=((0, 256), (0, 1), (0, 256), (0, 256), (0, 3), (0, 14), (0, 1), (0, 3), (0, 14), (0, 1)), contiguous=False), View(shape=(256, 1, 256, 256, 3, 31, 3, 31), strides=(462422016, 0, 1806336, 7056, 2352, 84, 28, 1), offset=0, mask=((0, 256), (0, 1), (0, 256), (0, 256), (0, 3), (0, 28), (0, 3), (0, 28)), contiguous=False), View(shape=(256, 1, 256, 256, 3, 30, 3, 30), strides=(566820864, 0, 8649, 2214144, 2790, 93, 30, 1), offset=0, mask=None, contiguous=False)))
    print(st)
    st_ = ShapeTracker(st.views[1:])
    print(*st.expr_idxs())
    print('asdf')
    print(*st_.expr_idxs())


if __name__ == '__main__':
  unittest.main()
