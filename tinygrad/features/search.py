from typing import Dict, List, cast, DefaultDict, Optional, Tuple
from tinygrad.lazy import vars_from_ast
from tinygrad.ops import Device, Compiled, MemBuffer
from tinygrad.helpers import prod, getenv, ImageDType, flatten, DEBUG
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.runtime.lib import RawBuffer
from collections import defaultdict
import functools, traceback
from tinygrad.shape.symbolic import sym_infer
from tqdm import tqdm
import multiprocessing as mp
import itertools

from tinygrad.codegen.optimizer import Opt, OptOps
actions = flatten([[Opt(op=OptOps.UPCAST, axis=axis, amt=amt) for amt in [0,2,3,4,7]] for axis in range(6)])
actions += flatten([[Opt(op=OptOps.UNROLL, axis=axis, amt=amt) for amt in [0,4]] for axis in range(4)])
actions += flatten([[Opt(op=OptOps.LOCAL, axis=axis, amt=amt) for amt in [2,3,4,8,16]] for axis in range(5)])
actions += [
  Opt(op=OptOps.LOCAL, axis=0, amt=32),
  Opt(op=OptOps.GROUP, axis=1, amt=4), Opt(op=OptOps.GROUP, axis=1, amt=8), Opt(op=OptOps.GROUP, axis=2, amt=8),
  Opt(op=OptOps.GROUPTOP, axis=0, amt=16), Opt(op=OptOps.GROUPTOP, axis=0, amt=256),
  Opt(op=OptOps.GROUPTOP, axis=1, amt=16), Opt(op=OptOps.GROUPTOP, axis=1, amt=256),
  Opt(op=OptOps.GROUPTOP, axis=2, amt=16), Opt(op=OptOps.GROUPTOP, axis=2, amt=256)
]

# returns time in seconds
import shelve
logtm = shelve.open(getenv("LOGTM", "")) if getenv("LOGTM", "") else None
def time_linearizer(lin:Linearizer, rawbufs:List[RawBuffer], allow_test_size=True, max_global_size=65536, cnt=3, should_copy=True, disable_cache=False) -> float:
  key = str((lin.ast, lin.applied_opts))
  if should_copy and not disable_cache and logtm is not None and key in logtm: return min(logtm[key])  # pylint: disable=E1135 # NOTE: we check should_copy since this may have side effects
  if should_copy: lin = lin.copy() # TODO: remove the need for this
  var_vals = {k:k.min for k in vars_from_ast(lin.ast)}
  try:
    lin.linearize()
    prg = cast(Compiled, Device[Device.DEFAULT]).to_program(lin)
    real_global_size = prg.global_size
    if allow_test_size and prg.global_size:
      test_global_size = prg.global_size[:]
      while prod(test_global_size) > max_global_size:
        for j in range(2,-1,-1):
          if test_global_size[j] > 16:
            test_global_size[j] //= 2
            break
      factor = prod(prg.global_size) / prod(test_global_size)
      prg.global_size = test_global_size
      #print(real_global_size, test_global_size, factor)
    else:
      factor = 1
    # TODO: this is super broken for var_vals
    # TODO: this is copied from prg.__call__
    global_size, local_size = prg.launch_dims(var_vals)
    if global_size is not None and local_size is None:
      local_size = prg.optimize_local_size(global_size, rawbufs)
      global_size = [g//l if g%l == 0 else g/l for g,l in zip(global_size, local_size)]
    tms = [prg.clprg(global_size, local_size, *rawbufs, *var_vals.values(), wait=True)*factor for _ in range(cnt)]
    prg.global_size = real_global_size
  except Exception:
    #import traceback; traceback.print_exc()
    #print("FAILED")
    #print(lin.ast)
    #print(lin.applied_opts)
    tms = [float('inf')]
  if logtm is not None: logtm[key] = tms
  return min(tms)

def run_and_time(prg, bufs, var_vals, baseline=None):
  first_tm = prg.exec(bufs, var_vals, force_wait=True, optimizing=True)
  if baseline is not None and baseline*5 < first_tm*1000: return first_tm*1000  # very slow
  return min([first_tm]+[prg.exec(bufs, var_vals, force_wait=True, optimizing=True) for _ in range(5)])*1000

# get (scrap) buffers for timing the linearizer
def bufs_from_lin(lin:Linearizer) -> List[RawBuffer]:
  bufsts:DefaultDict[int, List[MemBuffer]] = defaultdict(list)
  for x in lin.membufs: bufsts[x.idx].append(x)
  rawbufs:List[Optional[RawBuffer]] = [None]*len(bufsts)
  for k,lx in bufsts.items():
    rawbufs[k] = cast(Compiled, Device[Device.DEFAULT]).buffer(prod(lx[0].dtype.shape) if isinstance(lx[0].dtype, ImageDType) else max(y.st.size() for y in lx), lx[0].dtype)
  assert all(r is not None for r in rawbufs)
  return cast(List[RawBuffer], rawbufs)

def fresh_lin(lin:Linearizer):
  return apply_wino_upcast(Linearizer(lin.ast, lin.opts))

from tinygrad.ops import Device
def compile_kernel(k, checks=True):
  k.linearize()
  if checks: assert len(k.uops) < 2 ** 13, f"too many uops: {len(k.uops)}"  # device target compiler will take significantly longer than Linearizer
  prg = Device[Device.DEFAULT].to_program(k)
  return k.display_name, prg

def start_compile(pool, k, **kwargs):
  return pool.apply_async(compile_kernel, (k,), kwargs)

def catch_exception(f, on_fail=None):
  @functools.wraps(f)
  def inner(*args, **kwargs):
    try:
      return f(*args, **kwargs)
    except Exception:
      if DEBUG >= 3: traceback.print_exc()
      return on_fail
  return inner

def convert_parts(k, parts):
  result = []
  for axis, (local, upcast) in parts.items():
    axis_ = axis if axis < k.first_reduce else axis - k.first_reduce - len(k.group_for_reduce)
    if upcast > 1: result.append(Opt(OptOps.UNROLL if axis >= k.first_reduce else OptOps.UPCAST, axis_, upcast))
    if local > 1: result.append(Opt(OptOps.GROUP if axis >= k.first_reduce else OptOps.LOCAL, axis_, local))
  return result

def pick_beam(opts, beam_width, allow=getenv("BEAM_DIV", 2)):
  locals = {}
  seen = set()
  kept = []
  for lin, tm in opts:
    this_parts = tuple([x for x in lin.partitions.items()])
    if this_parts in seen: continue
    this_locals = tuple([(axis, local) for axis, (local, upcast) in lin.partitions.items()])
    if this_locals in locals and locals[this_locals] >= allow:
      continue
    locals[this_locals] = locals.get(this_locals, 0) + 1
    kept.append((lin, tm))
    seen.add(this_parts)
  return kept[:beam_width]

def apply_parts(fresh_k, a):
  k = fresh_k
  try:
    [k.apply_opt(op, simplify=False) for op in convert_parts(k, a)]
    k.partitions = a
    k.simplify_ones()
    assert prod(k.full_shape[k.shape_len - k.upcasted:k.shape_len]) <= 512, "too many upcasts!"
    assert prod(k.full_shape[k.first_reduce - k.local_dims:k.first_reduce + len(k.group_for_reduce)]) <= 256, "too many locals!"
    #assert prod(k.full_shape[k.first_reduce - k.local_dims:k.first_reduce + len(k.group_for_reduce)]) <= 128, "too many locals!"
    return k
  except Exception:
    return None

get_linearizer_actions_cache = {}
def get_next_parts(lin_parts, fresh_shape):
  old_parts = lin_parts
  #if (lin.ast, str(old_parts)) in get_linearizer_actions_cache:
  #  return get_linearizer_actions_cache[(lin.ast, str(old_parts))]
  actions = []

  ready = prod(prod(opt) for opt in old_parts.values()) >= 2 ** 4
  powers_of_two = [2, 4, 8, 16]
  if not ready:
    splits = powers_of_two
  else:
    splits = list(range(2, 32))

  for i, s in enumerate(fresh_shape):
    ol, ou = old_parts.get(i, (1, 1))
    for split in splits:
      if s % split != 0: continue
      if split in powers_of_two: actions.append(old_parts | {i: (ol * split, ou)})
      if split < 8: actions.append(old_parts | {i: (ol, ou * split)})
      if ol % split == 0:
        if split < 8: actions.append(old_parts | {i: (ol // split, ou * split)})
        actions.append(old_parts | {i: (ol // split, ou)})
      if ou % split == 0:
        if split in powers_of_two: actions.append(old_parts | {i: (ol * split, ou // split)})
        actions.append(old_parts | {i: (ol, ou // split)})
  for i, (ol, ou) in old_parts.items():
    for j, (ol2, ou2) in old_parts.items():
      if not (ol > 1 and ol2 > 1): continue
      if i >= j: continue
      swapped = list(old_parts.items())
      a, b = swapped.index((i, (ol, ou))), swapped.index((j, (ol2, ou2)))
      swapped[a], swapped[b] = (j, (ol2, ou2)), (i, (ol, ou))
      actions.append(dict(swapped))
  #get_linearizer_actions_cache[(lin.ast, str(old_parts))] = actions
  return actions

def get_linearizer_actions(lin:Linearizer, include_0=True):
  fresh = fresh_lin(lin)
  actions = get_next_parts(lin.partitions if hasattr(lin, "partitions") else {}, fresh.full_shape)

  acted_lins = {0:lin.copy()} if include_0 else {}
  for i,a in enumerate(actions):
    k = apply_parts(fresh_lin(lin), a)
    if k is not None: acted_lins[i + 1] = k
  return acted_lins

def apply_wino_upcast(self):
  # TODO: doing extra upcasts with images doesn't work for some reason (maybe has to do with to_image_idx)
  # to trigger the above bug, remove prod(self.full_shape[self.shape_len - self.upcasted:]) from the below
  # expression.
  # if there are small dims with lots of valid masks, upcast them (they might be from Tensor.stack)
  # this can be made much smarter
  to_upcast: List[int] = []
  # upcast leading axes first (hack-ish for winograd; we actually want to upcast masked axes with low stride first)
  for axis in range(self.first_reduce):
    # we might want to be able to split axes that are masked, or refuse to merge them in simplify_merge_adjacent
    # for now skip upcasting here if there is a symbolic axis
    if isinstance(self.full_shape[axis], int) and self.full_shape[axis] <= 7 and any(st.axis_is_masked(axis) for st in self.sts) and \
      prod(self.full_shape[self.shape_len - self.upcasted:]) * prod(self.full_shape[j] for j in to_upcast) * self.full_shape[axis] <= 7 * 7:
      if DEBUG >= 4: print(f"upcasting masked axis : {axis}")
      to_upcast.append(axis)
  for axis in to_upcast[::-1]:
    upcast_amt = self.full_shape[axis]
    assert isinstance(upcast_amt, int), "can only upcast ints"
    self.apply_opt(Opt(OptOps.UPCAST, axis, upcast_amt), simplify=False)
  return self

def beam_pass(pool, beam_time, best_lin, rawbufs, var_vals, greedy_i, beam_width):
  best_lin, best_time = best_lin
  acted_lins = itertools.chain(*(list(get_linearizer_actions(lin, include_0=False).items()) for lin, _ in beam_time))
  compiling = [(lin, start_compile(pool, lin)) for i, lin in acted_lins if i != 0]
  timed_lins = [(lin, catch_exception(lambda: run_and_time(prg.get(timeout=5)[1], rawbufs, var_vals, baseline=best_time), on_fail=float('inf'))()) for lin, prg in tqdm(compiling, desc=f'greedy layer {greedy_i}', disable=DEBUG != 1)]
  opts = sorted(timed_lins, key=lambda x: x[1])
  if opts and opts[0][1] <= best_time:
    best_lin, best_time = opts[0]
  beam_time = pick_beam(opts, beam_width)
  if DEBUG >= 1:
    gflops = sym_infer(best_lin.info.flops, var_vals) * 1e-9
    for kk, tm in beam_time[:4]:
      print(f"{tm:10.2f} ms ({gflops / tm * 1000:6.0f} gflops), from {len(opts):3d} actions", kk.colored_shape())
  return beam_time, (best_lin, best_time)



def beam_search(lin, rawbufs, beam_width):
  var_vals = {k:k.min for k in vars_from_ast(lin.ast)}

  lin = apply_wino_upcast(lin)
  if DEBUG >= 1: print(lin.full_shape)
  fresh = lin

  best_lin = (fresh, run_and_time(compile_kernel(fresh.copy())[1], rawbufs, var_vals))
  beam_time: List[Tuple[Linearizer, float]] = [best_lin]

  if best_lin[1] <= 0.05: return best_lin[0]  # skip if under 50us

  pool = mp.Pool(32)
  for greedy_i in range(16):
    beam_time, best_lin = beam_pass(pool, beam_time, best_lin, rawbufs, var_vals, greedy_i, beam_width)
    if not beam_time or beam_time[0][1] > best_lin[1]: break
  return best_lin[0]
