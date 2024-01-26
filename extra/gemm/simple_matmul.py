import numpy as np
from tinygrad.helpers import getenv
from tinygrad import dtypes, Tensor
from tinygrad.device import Device
dtype_in = dtypes.half if getenv("HALF") else dtypes.float
acc_dtype = dtypes.half if getenv("ACC_HALF") else None
N = getenv("N", 4096)
CNT = getenv("CNT", 10)
a, b = Tensor.rand(N, N, dtype=dtype_in).realize(), Tensor.rand(N, N, dtype=dtype_in).realize()
if (GPUS:=getenv("GPUS", 1)) > 1:
  device = tuple(f"{Device.DEFAULT}:{i}" for i in range(GPUS))
  a.shard_(device, -1)
  b.shard_(device, None)
for i in range(CNT):
  if i > 0 and getenv("RAND", 0) != 0:
    a, b = Tensor.rand(N, N, dtype=dtype_in).realize(), Tensor.rand(N, N, dtype=dtype_in).realize()
  c = a.matmul(b, acc_dtype=acc_dtype).realize()
  d = a.matmul(c, acc_dtype=acc_dtype).realize()
  print(a)
  print(b)
  print(c)
  print(d)
comp = a.numpy().astype(np.float32) @ b.numpy().astype(np.float32)
nc = c.numpy()
np.testing.assert_allclose(nc, comp, atol=1e-4, rtol=3e-2)
