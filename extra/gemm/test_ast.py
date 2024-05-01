from tinygrad import Tensor, Device, dtypes
from tinygrad.ops import LazyOp, BufferOps, UnaryOps, ReduceOps, BinaryOps, MemBuffer
from tinygrad.shape.shapetracker import ShapeTracker, View

ast = (LazyOp(op=BufferOps.STORE, src=(
  LazyOp(op=UnaryOps.CAST, src=(
    LazyOp(op=ReduceOps.SUM, src=(
      LazyOp(op=UnaryOps.CAST, src=(
        LazyOp(op=BinaryOps.MUL, src=(
          LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.half, st=ShapeTracker(views=(View(shape=(1, 256, 1, 128, 128, 1, 3, 58, 3, 58), strides=(0, 0, 0, 1, 1152, 0, 384, 0, 128, 0), offset=0, mask=None, contiguous=False),)))),
          LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=2, dtype=dtypes.half, st=ShapeTracker(views=(View(shape=(256, 128, 128, 3, 30, 2, 3, 30, 2), strides=(100352, 0, 784, 0, 28, 0, 0, 1, 0), offset=0, mask=((0, 256), (0, 128), (0, 128), (0, 3), (0, 28), (0, 1), (0, 3), (0, 28), (0, 1)), contiguous=False), View(shape=(256, 128, 128, 3, 61, 3, 61), strides=(530841600, 0, 32400, 0, 180, 60, 1), offset=-362, mask=((0, 256), (0, 128), (0, 128), (0, 3), (2, 61), (0, 3), (2, 61)), contiguous=False), View(shape=(1, 256, 1, 128, 128, 1, 3, 58, 3, 58), strides=(0, 548683776, 0, 0, 33489, 0, 10980, 183, 60, 1), offset=368, mask=None, contiguous=False)))))), arg=None),), arg=(dtypes.float, False)),), arg=(4, 6, 8)),), arg=(dtypes.half, False)),), arg=MemBuffer(idx=0, dtype=dtypes.half, st=ShapeTracker(views=(View(shape=(1, 256, 1, 128, 1, 1, 1, 58, 1, 58), strides=(0, 430592, 0, 3364, 0, 0, 0, 58, 0, 1), offset=0, mask=None, contiguous=True),)))),)
ast = (LazyOp(op=BufferOps.STORE, src=(LazyOp(op=UnaryOps.CAST, src=(LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=UnaryOps.CAST, src=(LazyOp(op=BinaryOps.MUL, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.half, st=ShapeTracker(views=(View(shape=(1, 256, 1, 128, 128, 1, 3, 58, 3, 58), strides=(0, 0, 0, 1, 1152, 0, 384, 0, 128, 0), offset=0, mask=None, contiguous=False),)))), LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=2, dtype=dtypes.half, st=
ShapeTracker(views=(View(shape=(256, 128, 128, 3, 30, 2, 3, 30, 2), strides=(100352, 0, 784, 0, 28, 0, 0, 1, 0), offset=0, mask=((0, 256), (0, 128), (0, 128), (0, 3), (0, 28), (0, 1), (0, 3), (0, 28), (0, 1)), contiguous=False), View(shape=(256, 128, 128, 3, 62, 3, 62), strides=(530841600, 0, 32400, 0, 180, 60, 1), offset=-543, mask=((0, 256), (0, 128), (0, 128), (0, 3), (3, 62), (0, 3), (3, 62)), contiguous=False), View(shape=(1, 256, 1, 128, 128, 1, 3, 58, 3, 58),
 strides=(0, 566820864, 0, 0, 34596, 0, 11346, 186, 61, 1), offset=561, mask=None, contiguous=False)))))), arg=None),), arg=(dtypes.float, False)),), arg=(4, 6, 8)),), arg=(dtypes.half, False)),), arg=MemBuffer(idx=0, dtype=dtypes.half
, st=ShapeTracker(views=(View(shape=(1, 256, 1, 128, 1, 1, 1, 58, 1, 58), strides=(0, 430592, 0, 3364, 0, 0, 0, 58, 0, 1), offset=0, mask=None, contiguous=True),)))),)

from tinygrad.codegen.linearizer import Linearizer
from tinygrad.codegen.kernel import Opt, OptOps
from tinygrad.features.search import bufs_from_lin

opts = [Opt(OptOps.UPCAST, 2, 2), Opt(OptOps.UPCAST, 3, 2), Opt(OptOps.UNROLL, 2, 3), Opt(OptOps.UNROLL, 1, 3), Opt(OptOps.TC, 0, 1)]

lin = Linearizer(*ast)
print(lin.full_shape)

for opt in opts:
  lin.apply_opt(opt)

lin.linearize()
print(lin.colored_shape())

for st in lin.sts: print(st)

prog = Device[Device.DEFAULT].to_program(lin)

bufs = bufs_from_lin(lin)
prog.exec(bufs, {})