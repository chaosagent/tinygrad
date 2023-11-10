from tinygrad.tensor import Tensor
from tinygrad.helpers import dtypes, getenv
from tinygrad.realize import run_schedule
from tinygrad.ops import Device

N = getenv("N", 2048)
BS = getenv("BS", 1)
ITER = getenv("ITER", 10)

a = Tensor.randn(BS, N, N, dtype=dtypes.half).realize()
b = Tensor.randn(BS, N, N, dtype=dtypes.half).realize()

c = (a.reshape((BS, N, 1, N)) * b.permute([0, 2, 1]).reshape((BS, 1, N, N))).cast(dtypes.float).sum(axis=-1).cast(dtypes.half)
s = c.lazydata.schedule()
si = s[-1]
for _ in range(ITER):
  Device[si.out.device].exec_ast(si.ast, output=si.out, inputs=si.inputs, var_vals=si.var_vals, **si.out._device_extra_args())