from typing import Dict, List, cast, DefaultDict, Optional, Tuple, Callable, Generator, Any
import itertools, functools, random, math, time, multiprocessing, traceback, signal
from collections import defaultdict
from tinygrad.device import Device, Compiled, Buffer, CompiledRunner, Compiler
from tinygrad.ops import MemBuffer
from tinygrad.helpers import prod, flatten, DEBUG, CACHELEVEL, diskcache_get, diskcache_put, getenv, Context, colored, to_function_name
from tinygrad.dtype import ImageDType
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.codegen.kernel import Opt, OptOps, KernelOptError
from tinygrad.tensor import Tensor
from tinygrad.shape.symbolic import sym_infer, Variable

actions = [Opt(op=OptOps.UPCAST, axis=axis, amt=amt) for amt in [0,2,3,4,5,7] for axis in range(6)]
actions += [Opt(op=OptOps.UNROLL, axis=axis, amt=amt) for amt in [0,4,7] for axis in range(4)]
actions += [Opt(op=OptOps.LOCAL, axis=axis, amt=amt) for amt in [2,3,4,8,13,16,29] for axis in range(5)]
actions += [Opt(op=OptOps.GROUPTOP, axis=axis, amt=amt) for amt in [13,16,28,29,32,49,64,256] for axis in range(3)]
actions += [Opt(op=OptOps.GROUP, axis=axis, amt=amt) for amt in [0,4,8,16] for axis in range(3)]
actions += [Opt(op=OptOps.PADTO, axis=axis, amt=amt) for amt in [32] for axis in range(7)]
actions += [Opt(op=OptOps.LOCAL, axis=0, amt=32), Opt(op=OptOps.UPCASTMID, axis=1, amt=4), Opt(op=OptOps.TC, axis=0, amt=0)]
actions += [Opt(op=OptOps.TC, axis=axis, amt=getenv("TC_OPT", 2)) for axis in range(4)]
if getenv("NOLOCALS"): actions += [Opt(op=OptOps.NOLOCALS)]

def _get_test_global_size(global_size, max_global_size, var_vals):
  test_global_size, factor = [sym_infer(sz, var_vals) for sz in global_size], 1
  while prod(test_global_size) > max_global_size:
    for j in range(len(global_size)-1,-1,-1):
      if test_global_size[j] > 16:
        test_global_size[j] //= 2
        factor *= 2
        break
  return test_global_size, factor

def _time_program(variables:List[Variable], outcount:int, rdev:Compiled, lib:bytes, global_size, local_size, var_vals, rawbufs,
                  early_stop=None, max_global_size=65536, clear_l2=False, cnt=3, name="test"):
  factor = 1
  if global_size is not None and max_global_size is not None:
    global_size, factor = _get_test_global_size(global_size, max_global_size, var_vals)
  try: car = CompiledRunner(name, "", rdev.dname, global_size, local_size, variables=variables, precompiled=lib, outcount=outcount)
  except AssertionError: return [math.inf] * cnt
  tms = []
  for _ in range(cnt):
    if clear_l2:
      with Context(DEBUG=0, BEAM=0, CACHECOLLECTING=0): Tensor.ones(1024,1024).contiguous().realize()
    tms.append(cast(float, car(rawbufs, var_vals, wait=True))*factor)
    if early_stop is not None and early_stop < tms[-1]: break
  return tms

def _compile_linearizer(compiler:Compiler, lin:Linearizer, name:Optional[str]=None, enforce_max:bool=False) \
  -> Tuple[bytes, Optional[List[int]], Optional[List[int]], List[Variable], int, float, int]:
  lin.linearize()
  if enforce_max and len(lin.uops.uops) >= getenv("BEAM_UOPS_MAX", 3000) > 0: raise RuntimeError("too many uops")
  src = compiler.render(name if name is not None else to_function_name(lin.name), lin.uops)   # NOTE: these all have the same name for deduping
  if DEBUG >= 5: print(src)
  st = time.perf_counter()
  prog = compiler.compile(src)
  et = time.perf_counter() - st
  return prog, lin.global_size, lin.local_size, lin.uops.vars(), len(lin.outbufs), et, len(lin.uops.uops)

def _try_compile_linearized_w_idx(x:Tuple[Any,Linearizer, Compiler]):
  try: return x[0], _compile_linearizer(x[2], x[1], "test", enforce_max=True)
  except Exception:
    if DEBUG >= 4: traceback.print_exc()
    return x[0], None

# workers should ignore ctrl c
def _init_worker(): signal.signal(signal.SIGINT, signal.SIG_IGN)

def _ensure_buffer_alloc(bufs:List[Buffer]) -> List[Buffer]: return [buf.ensure_allocated() for buf in bufs]

# *** external API ***

# get (scrap) buffers for timing the linearizer
def bufs_from_lin(lin:Linearizer, allocate:bool=True) -> List[Buffer]:
  bufsts:DefaultDict[int, List[MemBuffer]] = defaultdict(list)
  for x in lin.membufs: bufsts[x.idx].append(x)
  rawbufs:List[Optional[Buffer]] = [None]*len(bufsts)
  for k,lx in bufsts.items():
    buf_size = prod(lx[0].dtype.shape) if isinstance(lx[0].dtype, ImageDType) else max(y.st.real_size() for y in lx)
    if buf_size == 0: buf_size = 1  # create a size 1 buffer if no cell is accessed in kernel. # TODO: remove from kernel input in this case.
    rawbufs[k] = Buffer(lin.opts.device, buf_size, lx[0].dtype).allocate() if allocate else Buffer(lin.opts.device, buf_size, lx[0].dtype)
  assert all(r is not None for r in rawbufs)
  return cast(List[Buffer], rawbufs)

# get dictionary of all possible actions
def get_linearizer_actions(lin:Linearizer, include_0=True) -> Dict[int, Linearizer]:
  acted_lins, max_up, max_lcl = {0:lin} if include_0 else {}, getenv("BEAM_UPCAST_MAX", 256), getenv("BEAM_LOCAL_MAX", 256)
  for i,a in enumerate(actions):
    if a.axis is not None and a.axis >= lin.shape_len: continue
    if a.axis is not None and lin.full_shape[a.axis] == a.amt and Opt(a.op, a.axis, 0) in actions: continue
    lin2 = lin.copy()
    try:
      lin2.apply_opt(a)
      up, lcl, tc_up = 1, 1, prod(tc.dims)//prod([x[1] for x in tc.threads]) if (tc:=lin2.tensor_core) else 1
      for s,c in zip(lin2.full_shape, lin2.colors()):
        if c in {"magenta", "yellow"}: up *= s
        elif c in {"cyan", "green", "white"}: lcl *= s
      if up//tc_up > max_up or lcl > max_lcl: continue
      acted_lins[i+1] = lin2
    except KernelOptError: pass
  return acted_lins

beam_pool = None
def _beam_search(lin:Linearizer, rawbufs:List[Buffer], amt:int, allow_test_size=True) -> Generator[Linearizer, List[Tuple[Linearizer, float]], Linearizer]:
  key = {"ast": lin.ast[0].key, "amt": amt, "allow_test_size": allow_test_size, "device": lin.opts.device, "suffix": lin.opts.suffix}
  if (val:=diskcache_get("beam_search", key)) is not None and not getenv("IGNORE_BEAM_CACHE") and CACHELEVEL >= 1:
    ret = lin.copy()
    for o in val[len(lin.applied_opts):]: ret.apply_opt(o)
    return ret

  beam: List[Tuple[Linearizer, float]] = []
  min_progress_micros = getenv("BEAM_MIN_PROGRESS", 0.01)

  rawbufs = _ensure_buffer_alloc(rawbufs)
  var_vals = {k:(k.max+k.min)//2 for k in lin.ast[0].vars()}
  exiting, st = False, time.perf_counter()
  dev = Device[lin.opts.device]
  while not exiting:
    acted_lins: List[Linearizer] = flatten([get_linearizer_actions(lin, include_0=False).values() for lin,_ in beam]) if len(beam) else [lin]
    early_stop = beam[0][1] * 3 if len(beam) else 1.0
    timed_lins = yield acted_lins, rawbufs, var_vals, dev, early_stop

    # done
    opts = sorted(timed_lins, key=lambda x: x[1])
    exiting = len(opts) == 0 or (len(beam) > 0 and ((beam[0][1]-opts[0][1])*1e6 < min_progress_micros))
    if not exiting: beam = opts[:amt]
    elif len(opts) > 0 and opts[0][1] < beam[0][1]: beam = opts[:1]
    assert len(beam) > 0, "no BEAM items succeeded?!?" # this asserts in unet3d multi-gpu, need to figure out why
    if DEBUG >= 2: print(f"\r{time.perf_counter() - st:7.2f}s:", colored(f"{beam[0][1]*1e6:12.2f} us", "green" if exiting else None), f"from {len(acted_lins):3d} -> {len(opts):3d} actions\033[K", beam[0][0].colored_shape())  # noqa: E501

  if CACHELEVEL >= 1: diskcache_put("beam_search", key, beam[0][0].applied_opts)
  return beam[0][0]

class GeneratorWrapper:
  def __init__(self, gen): self.gen = gen

  def send(self, x): self.gen.send(x)
  def __iter__(self): self.value = yield from self.gen
def async_compile_engine(coros, allow_parallel=True) -> Generator[Any, None, None]:
  global beam_pool
  from collections import deque

  #default_parallel = multiprocessing.cpu_count() if all([lin.opts.device in {"CUDA", "HSA", "AMD", "NV"} for lin, _ in lin_rawbufs]) else 0
  default_parallel = multiprocessing.cpu_count() if allow_parallel else 0
  PARALLEL = getenv("parallel", default_parallel)
  if beam_pool is None and PARALLEL:
    beam_pool = multiprocessing.get_context("spawn").Pool(PARALLEL, _init_worker, (), getenv("BEAM_MAX_TASKS_PER_CHILD", 16))

  try:
    seen_libs = set()
    q = deque(coros)
    working = []
    working_timed = []
    completed = []
    while q or working:
      while q and len(working) < max(PARALLEL, 1):
        new_coro = q.popleft()
        working.append(new_coro)
        working_timed.append(None)
        completed.append(None)

      st = time.perf_counter()

      compile_jobs = {}
      for job_i, (search_coro, timed_lins) in enumerate(zip(working, working_timed)):
        try:
          acted_lins, rawbufs, var_vals, dev, early_stop = search_coro.send(timed_lins)
        except StopIteration as e:
          completed[job_i] = e.value
          continue
        for lin_i, acted_lin in enumerate(acted_lins): compile_jobs[job_i, lin_i] = (acted_lin, rawbufs, var_vals, dev, early_stop)
        working_timed[job_i] = []

      _compile_fn = _try_compile_linearized_w_idx
      mapper = map if beam_pool is None else beam_pool.imap_unordered
      for (job_i, lin_i), proc in mapper(_compile_fn, [((job_i, lin_i), acted_lin, dev.compiler) for (job_i, lin_i), (acted_lin, _, _, dev, _) in compile_jobs.items()]):
        if proc is None: continue
        lib, global_size, local_size, vars, outcount, compile_et, num_uops = proc
        if lib in seen_libs: continue
        # print(acted_lins[i].colored_shape(), acted_lins[i].applied_opts)  # for debugging BEAMs that segfault
        seen_libs.add(lib)
        acted_lin, rawbufs, var_vals, dev, early_stop = compile_jobs[job_i, lin_i]
        try:
          tms = _time_program(vars, outcount, dev, lib, global_size, local_size, var_vals, rawbufs, early_stop=early_stop)
        except RuntimeError:
          continue  # for runtime issues
        working_timed[job_i].append((acted_lin, min(tms)))
        if getenv("BEAM_LOG") > 0:
          print(
            f"{time.perf_counter() - st:7.2f}s: {lin_i:5d} {num_uops:5d} uops {compile_et * 1e6:12.2f} us compile/{min(tms) * 1e6:12.2f} us run       {len(working_timed[job_i]):4d}/{len(compile_jobs):4d}         {acted_lin.colored_shape()}")  # noqa: E501
        elif DEBUG >= 2:
          print(f"\r{time.perf_counter() - st:7.2f}s: {min(tms) * 1e6:12.2f} us       {len(working_timed[job_i]):4d}/{len(compile_jobs):4d}         {acted_lin.colored_shape()}\033[K", end="")  # noqa: E501

      if completed[0] is not None:
        working.pop(0)
        working_timed.pop(0)
        yield completed.pop(0)
  except KeyboardInterrupt as e:
    if beam_pool is not None: beam_pool.terminate()
    beam_pool = None
    raise e

def optimize_local_size(clprg:Callable, global_size:List[int], rawbufs:List[Buffer]) -> List[int]:
  test_rawbuffers = [Buffer(rawbufs[0].device, rawbufs[0].size, rawbufs[0].dtype).allocate(), *rawbufs[1:]] if rawbufs[0] in rawbufs[1:] else rawbufs
  MAX_WORKGROUP = 1024
  local_dims = [[x for x in set([sz, 1, 2, 4, 8, 16, 32, 64, 128, 256, MAX_WORKGROUP]) if x<=sz] for sz in global_size]
  local_sizes = [list(x) for x in itertools.product(*local_dims) if prod(x) <= MAX_WORKGROUP] * 2  # try each valid size twice
  def try_exec(local_size):
    try: return clprg(*[x._buf for x in test_rawbuffers], global_size=[g//l if g%l == 0 else g/l for g,l in zip(global_size, local_size)], local_size=local_size, wait=True)  # noqa: E501
    except Exception: return float('inf')
  ret = min([(try_exec(local_size), local_size) for local_size in random.sample(local_sizes, len(local_sizes))])
  assert not math.isinf(ret[0]), "all optimize_local_size exec failed"
  return ret[1]

def time_linearizer(lin:Linearizer, rawbufs:List[Buffer], allow_test_size=True, max_global_size=65536, cnt=3, disable_cache=False, clear_l2=False) -> float:  # noqa: E501
  key = {"ast": lin.ast[0].key, "opts": str(lin.applied_opts), "allow_test_size": allow_test_size, "max_global_size": max_global_size, "clear_l2": clear_l2, "device": lin.opts.device, "suffix": lin.opts.suffix}  # noqa: E501
  if not disable_cache and CACHELEVEL >= 2 and (val:=diskcache_get("time_linearizer", key)) is not None: return min(val)

  dev = Device[lin.opts.device]
  assert dev.compiler is not None

  rawbufs = _ensure_buffer_alloc(rawbufs)
  var_vals = {k:(k.max+k.min)//2 for k in lin.ast[0].vars()}
  lib, global_size, local_size, vars, outcount, _, _ = _compile_linearizer(dev.compiler, lin)
  tms = _time_program(vars, outcount, dev, lib, global_size, local_size, var_vals, rawbufs, max_global_size=max_global_size if allow_test_size else None, clear_l2=clear_l2, cnt=cnt, name=to_function_name(lin.name))  # noqa: E501

  if CACHELEVEL >= 2: diskcache_put("time_linearizer", key, tms)
  return min(tms)
