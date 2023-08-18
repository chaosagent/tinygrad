# to start thinking about the $2,000 norm fusion bounty

from tinygrad.tensor import Tensor
from tinygrad.nn import Conv2d, BatchNorm2d
from tinygrad.state import get_parameters
from tinygrad.helpers import dtypes

if __name__ == "__main__":
  Tensor.training = True

  BS, C1, H, W = 4, 128, 224, 224
  C2, K, S, P = 128, 3, 1, 1

  Tensor.default_type = dtypes.half
  x = Tensor.uniform(BS, C1, H, W)
  conv = Conv2d(C1, C2, kernel_size=K, stride=S, padding=P)
  bn = BatchNorm2d(C2, track_running_stats=False)
  for t in get_parameters([x, conv, bn]): t.realize()

  print("running network")
  x.sequential([conv, bn]).numpy()
