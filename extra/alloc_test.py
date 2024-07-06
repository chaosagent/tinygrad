from tinygrad import dtypes
from tinygrad.device import Buffer, Device
import traceback

l = 0
r = 32 * 2 ** 30
while l + 1 != r:
  m = l + (r - l) // 2
  try:
    b = Buffer(Device.DEFAULT, m, dtypes.uint8).allocate()
    del b
    l = m
    print(hex(m), 'success')
  except:
    r = m
    print(hex(m), 'fail')
print('largest allocated:', hex(l))