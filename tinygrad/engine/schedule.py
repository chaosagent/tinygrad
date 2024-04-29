import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional, Set, DefaultDict
from tinygrad.ops import LoadOps, ScheduleItem, BufferOps, LazyOp, ReduceOps, ConstBuffer, MemBuffer, BinaryOps, UnaryOps
from tinygrad.features.graph import log_lazybuffer, realized_lazybuffer
from tinygrad.helpers import GRAPH, DEBUG, GlobalCounters, prod, dedup, all_int, getenv
from tinygrad.shape.symbolic import Variable
from tinygrad.dtype import ImageDType, dtypes
from tinygrad.lazy import LazyBuffer
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View

# creation can recurse a lot
sys.setrecursionlimit(10000)

# TODO: it's unfortunate this needs to exist, but because of ASSIGN, we have to retain the LazyBuffer structure until post toposort
@dataclass(frozen=True)
class _LBScheduleItem:
  ast: Tuple[LazyOp, ...]
  outputs: Tuple[LazyBuffer, ...]
  inputs: Tuple[LazyBuffer, ...]
  var_vals: Dict[Variable, int]

# recursively create a lazyop
def _recursive_lazyop(buf:LazyBuffer, membufs:List[LazyBuffer], var_vals:Dict[Variable, int], st:ShapeTracker,
                      realizes:Set[LazyBuffer], cache, first=True, assign_to:Optional[LazyBuffer]=None, assign_idx:Optional[int]=None) -> LazyOp:
  if (buf, st) in cache: return cache[(buf, st)]
  if buf != buf.base:
    st = buf.st + st
    buf = buf.base
  # all buffers here are base now
  assert buf.op is not None

  # consts are always fused and generated
  if buf.op is LoadOps.CONST:
    unbound_st, st_var_vals = st.simplify().unbind()
    var_vals.update(st_var_vals)
    return LazyOp(BufferOps.CONST, (), ConstBuffer(buf.arg, buf.dtype, unbound_st))

  # if we aren't fusing it, it's a load and we add it to the inputs
  if buf.realized or (buf in realizes and not first):
    unbound_st, st_var_vals = st.simplify().unbind()
    var_vals.update(st_var_vals)
    if assign_to is not None and buf is assign_to:
      assert assign_idx is not None
      if not unbound_st.contiguous:
        # we also allow masked views. if it has a single view and it's equal when you shrink a contig, it's fine
        if not (len(unbound_st.views) == 1 and unbound_st.views[0].mask is not None and
            ShapeTracker.from_shape(unbound_st.shape).shrink(unbound_st.views[0].mask) == unbound_st.shrink(unbound_st.views[0].mask)):
          raise RuntimeError(f"must be contiguous for assign {unbound_st}")
      return LazyOp(BufferOps.LOAD, (), MemBuffer(assign_idx, buf.dtype, unbound_st))
    if buf not in membufs: membufs.append(buf)
    return LazyOp(BufferOps.LOAD, (), MemBuffer(membufs.index(buf), buf.dtype, unbound_st))

  # if a CONTIGUOUS or ASSIGN made it all the way here, just skip it
  if buf.op is LoadOps.CONTIGUOUS:
    assert first
    return _recursive_lazyop(buf.srcs[0], membufs, var_vals, st, realizes, cache, False)
  if buf.op is LoadOps.ASSIGN:
    assert first
    assert buf.srcs[1].base is buf.srcs[1], "assign must be to base"
    assert buf.srcs[1].realized is not None, f"assign must be already realized to schedule {buf.srcs[1]}"
    return _recursive_lazyop(buf.srcs[0], membufs, var_vals, st, realizes, cache, False, assign_to=buf.srcs[1], assign_idx=membufs.index(buf))

  # if it's a reduce, we have to change the shapetracker
  if buf.op in ReduceOps:
    assert st.contiguous, "ReduceOps late fusion must be contiguous"
    st = ShapeTracker.from_shape(buf.srcs[0].shape)

  if buf.op is BinaryOps.MUL and getenv("MUL_MASK_OPT", 1):
    src_ops = tuple(_recursive_lazyop(x, membufs, var_vals, ShapeTracker.from_shape(st.shape), realizes, cache, False, assign_to, assign_idx) for x in buf.srcs)
    if not any(lop.op in ReduceOps for src_op in src_ops for lop in src_op.lazyops):
      from tinygrad.features.graph import print_tree
      src_sts = [[lop.arg.st for lop in src_op.lazyops if lop.op in [BufferOps.LOAD, BufferOps.CONST]] for src_op in src_ops]
      mangled_sts = [ShapeTracker.from_shape(st.shape) for _ in buf.srcs]
      if DEBUG >= 4: print(st)
      for view in st.views:
        if DEBUG >= 4: print('view', view)
        stride_part = View.create(view.shape, strides=view.strides, offset=view.offset, mask=None)
        view_mask = view.mask or tuple((0, s) for s in view.shape)
        unmasked_stride_part = View.create(tuple(b - a for a, b in view_mask), strides=view.strides, offset=view.offset + sum([a * strd for (a, b), strd in zip(view_mask, view.strides)]), mask=None)
        src_zeros = []
        for src_op_sts in src_sts:
          zeros = None
          for src_st in src_op_sts:
            src_st = src_st + ShapeTracker((unmasked_stride_part,))
            if DEBUG >= 4: print('src_st', src_st)
            src_real_strides = src_st.real_strides()
            if DEBUG >= 4: print('src_real_strides', src_st.real_strides())
            this_zeros = set([i for i, x in enumerate(src_real_strides) if x == 0])
            if zeros is None:
              zeros = this_zeros
            else:
              zeros = zeros.intersection(this_zeros)
          src_zeros.append(zeros)
        mask_assignment = [[] for _ in buf.srcs]
        for i in range(len(view.shape)):
          if i in src_zeros[0]: mask_assignment[1].append(i)
          elif i in src_zeros[1]: mask_assignment[0].append(i)
          else:
            mask_assignment[0].append(i)
            mask_assignment[1].append(i)
        if DEBUG >= 4: print('mask_assignment[0]', mask_assignment[0])
        if DEBUG >= 4: print('mask_assignment[1]', mask_assignment[1])
        if view.mask is not None:
          if DEBUG >= 4: print('stride part', stride_part)
          mask_part0 = tuple((m if i in mask_assignment[0] else (0, s)) for i, (m, s) in enumerate(zip(view.mask, view.shape)))
          mask_part1 = tuple((m if i in mask_assignment[1] else (0, s)) for i, (m, s) in enumerate(zip(view.mask, view.shape)))
          if DEBUG >= 4: print('mask_part0', mask_part0)
          if DEBUG >= 4: print('mask_part1', mask_part1)
        else:
          mask_part0 = tuple([(0, s) for s in view.shape])
          mask_part1 = tuple([(0, s) for s in view.shape])
        for src_i, src_op_sts in enumerate(src_sts):
          st_update = ShapeTracker((View.create(view.shape, strides=tuple((strd if i in mask_assignment[src_i] else 0) for i, strd in enumerate(view.strides)),
                                                offset=view.offset + sum([0 if i in mask_assignment[src_i] else a * strd for i, ((a, b), strd) in enumerate(zip(view_mask, view.strides))]),
                                                mask=mask_part0 if src_i == 0 else mask_part1),))
          if DEBUG >= 4: print(f'st_update[{src_i}]', st_update)
          mangled_sts[src_i] = mangled_sts[src_i] + st_update
          if DEBUG >= 4: print(f'mangled_sts[{src_i}]', mangled_sts[src_i])
          for st_i, src_st in enumerate(src_op_sts):
            src_op_sts[st_i] = src_st + st_update
      final_src_ops = tuple(_recursive_lazyop(x, membufs, var_vals, mangled_sts[src_i], realizes, cache, False, assign_to, assign_idx) for src_i, x in enumerate(buf.srcs))
      cache[(buf, st)] = ret = LazyOp(buf.op, final_src_ops, buf.arg).simplify()
      if DEBUG >= 4: print_tree(ret)
      return ret

  # otherwise we fuse it like normal
  cache[(buf, st)] = ret = \
    LazyOp(buf.op, tuple(_recursive_lazyop(x, membufs, var_vals, st, realizes, cache, False, assign_to, assign_idx) for x in buf.srcs), buf.arg).simplify()
  return ret

def _schedule_one(out:LazyBuffer, realizes:Set[LazyBuffer], reduce_for_op: Dict[LazyBuffer, LazyBuffer]) -> _LBScheduleItem:
  inputs: List[LazyBuffer] = []
  var_vals: Dict[Variable, int] = out.st.var_vals.copy()
  if out.op in {LoadOps.CUSTOM, LoadOps.COPY, LoadOps.EMPTY}:
    op, inputs = LazyOp(out.op, (), out.arg), list(out.srcs)
  else:
    output_st, membufs = ShapeTracker.from_shape(reduce_for_op[out].shape if out in reduce_for_op else out.shape), [out]
    output_view = out.arg[0] if out.op is LoadOps.ASSIGN and out.arg else output_st
    op = _recursive_lazyop(out, membufs, var_vals, output_st, realizes, cache={})
    output_view, vv = output_view.simplify().unbind()
    if vv: var_vals.update(vv)
    op, inputs = LazyOp(BufferOps.STORE, (op, ), MemBuffer(0, out.dtype, output_view)), membufs[1:]
  return _LBScheduleItem((op,), (out,), tuple(inputs), var_vals)

# recursively search the entire graph for all LazyBuffers, insert realizes after expands
def _recurse_lb(buf:LazyBuffer, realizes:Set[LazyBuffer], allbufs:Dict[LazyBuffer, None],
                simple_pads:Set[LazyBuffer], children:DefaultDict[LazyBuffer, Dict[LazyBuffer, None]], scheduled=False):
  if buf in allbufs or buf.base.realized: return
  if GRAPH: log_lazybuffer(buf, scheduled)
  if isinstance(buf.dtype, ImageDType) and (prod(buf.shape) != prod(buf.dtype.shape) or
                                            not any(buf.shape[x]%4 == 0 for x in buf.st.unit_stride_axes())):
    if DEBUG >= 3: print(f"forcing image {buf.dtype} with shape {buf.shape} to float32")
    buf.dtype = dtypes.float32  # NOTE: this is what makes the dtype above not match
    # hack the underlying buffer too
    if buf.base is buf:
      assert not hasattr(buf.buffer, '_buf'), "can't fixup allocated buffer"
      buf.buffer.dtype = dtypes.float32
      buf.buffer.options = None
  if buf.base != buf:
    if (base_sz := prod(buf.base.st.shape)) < prod(buf.st.shape):
      realize_expand = False
      if all_int(buf.base.st.shape):
        for view in buf.st.views:
          # theoretically we can realize before shrink and expands that happen in the same st
          if not all_int(view.shape):
            realize_expand = True
            break
          if prod(view.shape) > base_sz:
            #print(prod(buf.base.st.shape), base_sz, buf.base.st.shape, view.shape)
            if not (view.mask and prod([y - x for x, y in view.mask]) <= base_sz):
              #print(f'bad {buf.base.st.shape} {view.mask}')
              realize_expand = True
              break
            base_sz = prod(view.shape)
        else:
          #print('good')
          #print(buf.base.op)
          simple_pads.add(buf.base)
      if realize_expand:
        if buf not in realizes and buf.base.op == UnaryOps.CAST and buf.base.dtype.itemsize >= buf.base.srcs[0].base.dtype.itemsize:
          realizes.add(buf.base.srcs[0].base)
        else:
          realizes.add(buf.base)
    return _recurse_lb(buf.base, realizes, allbufs, simple_pads, children)
  if buf.forced_realize: realizes.add(buf)
  allbufs[buf] = None
  if buf.op in LoadOps: realizes.add(buf.base)
  if buf.op is LoadOps.COPY:
    assert buf.srcs[0].st.contiguous and buf.srcs[0].size == buf.srcs[0].base.size, "can only copy contig"
    realizes.add(buf.srcs[0].base)
  for x in buf.srcs:
    children[x.base][buf] = None
    _recurse_lb(x, realizes, allbufs, simple_pads, children)

UNSAFE_PAD_OPS = {BinaryOps.DIV, BinaryOps.CMPLT, BinaryOps.CMPEQ, UnaryOps.LOG2, UnaryOps.EXP2}
def _is_padding_okay(buf:LazyBuffer, realizes:Set[LazyBuffer]) -> bool:
  if buf in realizes or buf.realized: return True
  # NOTE: this broke to_image_idx and coder with JIT
  if buf.op in UNSAFE_PAD_OPS: return False
  return all(_is_padding_okay(x.base, realizes) for x in buf.srcs)

def create_schedule(outs:List[LazyBuffer], seen:Optional[Set[LazyBuffer]]=None) -> List[ScheduleItem]:
  if seen is None: seen = set()

  # start by just realizing the buffers passed in
  realizes: Set[LazyBuffer] = set([x.base for x in outs if not x.base.realized])
  allbufs: Dict[LazyBuffer, None] = {}
  simple_pads: Set[LazyBuffer] = set()
  children: DefaultDict[LazyBuffer, Dict[LazyBuffer, None]] = defaultdict(dict)
  for out in outs: _recurse_lb(out.base, realizes, allbufs, simple_pads, children, scheduled=True)

  # check if we have to realize pads
  for p in simple_pads:
    if not _is_padding_okay(p, realizes):
      realizes.add(p)

  # find all reduces, and pair them to a elementwise op. if they can't be cleanly paired, force realize the reduce (or a contig child)
  reduce_for_op: Dict[LazyBuffer, LazyBuffer] = {}
  for r in allbufs.keys():
    if r != r.base or r.op not in ReduceOps or r in realizes: continue

    # follow the reduce down
    child_set: Dict[LazyBuffer, ShapeTracker] = {r: r.st}
    realized_children: Dict[LazyBuffer, ShapeTracker] = {}
    forced_realize = False
    can_chase = True
    while not forced_realize and len(child_set):
      next_child_set = {}
      for tr,st in child_set.items():
        if tr in realizes:
          realized_children[tr] = st
          # can only have one output buffer
          # can only reduce contiguous
          # max one reduceop per kernel
          if len(realized_children) > 1 or not st.contiguous or st.size != r.st.size or (tr in reduce_for_op and reduce_for_op[tr] != r):
            can_chase = tr not in reduce_for_op or reduce_for_op[tr] == r
            forced_realize = True
            break
          continue
        for tr_next in children[tr].keys():
          if not tr_next.realized:
            # max one reduceop per kernel
            if tr_next.op in ReduceOps:
              forced_realize = True
              break
            st_childs = dedup([s for s in tr_next.srcs if s.base == tr])
            if len(st_childs) > 1:
              forced_realize = True
              break
            next_child_set[tr_next] = st + st_childs[0].st
      child_set = next_child_set
    if forced_realize:
      tr = r
      if can_chase:
        # can chase this down to contiguous children
        st = tr.st
        while len(children[tr]) == 1:
          tr_next = next(iter(children[tr].keys()))
          st_childs = dedup([s for s in tr_next.srcs if s.base == tr])
          if len(st_childs) > 1: break
          if st.size != st_childs[0].st.size: break
          st = st + st_childs[0].st
          if not st.contiguous or tr_next.op in ReduceOps: break
          tr = tr_next
        # don't cast to higher size before store
        if tr.op is UnaryOps.CAST and tr.arg[0].itemsize > tr.srcs[0].dtype.itemsize:
          tr = tr.srcs[0].base
        reduce_for_op[tr] = r
      realizes.add(tr)
    else:
      assert len(realized_children) == 1
      reduce_for_op[next(iter(realized_children.keys()))] = r

  # preschedule all buffers in realizes
  prescheduled = {x:_schedule_one(x, realizes, reduce_for_op) for x in realizes if x not in seen and x.realized is None and x.op is not LoadOps.CONST}
  assign_targets = {x.srcs[1]:x for x in realizes if x.op is LoadOps.ASSIGN and x not in seen and x.realized is None}

  # breadth first ordering
  graph: DefaultDict[LazyBuffer, List[LazyBuffer]] = defaultdict(list)
  in_degree: DefaultDict[LazyBuffer, int] = defaultdict(int)
  for key, si in prescheduled.items():
    for x in si.inputs:
      graph[x].append(key)
      if x in assign_targets:
        graph[key].append(assign_targets[x])
        in_degree[assign_targets[x]] += 1
      if x in prescheduled: in_degree[key] += 1
    for out in si.outputs: del out.srcs  # can only schedule once

  queue = deque(si for key, si in prescheduled.items() if in_degree[key] == 0)
  schedule: List[ScheduleItem] = []
  kernel_number = GlobalCounters.kernel_count
  while queue:
    ps = queue.popleft()
    for buf in ps.outputs: seen.add(buf)
    if GRAPH:
      kernel_number += 1
      for out in ps.outputs: realized_lazybuffer(out, kernel_number)
    schedule.append(ScheduleItem(ps.ast, tuple(x.buffer for x in ps.outputs if x.size != 0),
                                 tuple(x.buffer for x in ps.inputs if x.size != 0), ps.var_vals))
    for x in graph[ps.outputs[0]]:
      in_degree[x] -= 1
      if in_degree[x] == 0: queue.append(prescheduled[x])

  # confirm everything was scheduled correctly
  if not all(degree == 0 for degree in in_degree.values()) or len(prescheduled) != len(schedule):
    raise RuntimeError(f"cycle detected in graph, prescheduled {len(prescheduled)} but only scheduled {len(schedule)}")
  return schedule
