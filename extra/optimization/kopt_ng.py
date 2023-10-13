from typing import Callable, Dict, Tuple, List, Any, cast
import time
import traceback
import multiprocessing as mp
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.codegen.optimizer import OptOps, Opt
from tinygrad.codegen.search import run_and_time
from tinygrad.helpers import DEBUG, prod, getenv, ansilen
from tinygrad.lazy import vars_from_ast

from extra.optimization.helpers import start_compile

def get_divisors(n, min_div = 1, max_div = 512, extra=None):
  if min_div > 1: yield 1
  if extra is not None and extra < min_div and extra != 1 and n % extra == 0: yield extra
  for d in range(min_div, max_div + 1):
    if n % d == 0: yield d
  if extra is not None and extra > max_div and extra != 1 and n % extra == 0: yield extra

def kernel_optimize_opts(k:Linearizer, suggestion):
  import nevergrad as ng
  suggestion = suggestion_to_dict(suggestion)
  opts = []
  for i in range(k.first_reduce):
    # TODO: the upcast always happen first, you might want to reverse this?
    # TODO: the order of the locals might improve things too
    opts.append([Opt(OptOps.UPCAST, i, s) for s in get_divisors(k.full_shape[i], max_div=8, extra=suggestion.get((OptOps.UPCAST, i)))])
    opts.append([Opt(OptOps.LOCAL, i, s) for s in get_divisors(k.full_shape[i], min_div=4, extra=suggestion.get((OptOps.LOCAL, i)))])
  for i in range(k.first_reduce, k.shape_len):
    opts.append([Opt(OptOps.UPCAST, i, s) for s in get_divisors(k.full_shape[i], max_div=8, extra=suggestion.get((OptOps.UPCAST, i)))])
    opts.append([Opt(OptOps.GROUP, i, s) for s in get_divisors(k.full_shape[i], min_div=4, extra=suggestion.get((OptOps.GROUP, i))) if all(st.shape[i] % s == 0 or st.shape[i] == 1 for st in k.sts)])
    opts.append([Opt(OptOps.GROUPTOP, i, s) for s in get_divisors(k.full_shape[i], min_div=4, extra=suggestion.get((OptOps.GROUPTOP, i))) if all(st.shape[i] % s == 0 or st.shape[i] == 1 for st in k.sts)])
  # nevergrad parameters, default parameter choice
  return [ng.p.TransitionChoice(opt) for opt in opts], [opt[0] for opt in opts]

def suggestion_to_dict(suggestion):
  result: Dict[Tuple[OptOps, int], int] = {}
  # this is lossy
  for op in (suggestion or []): result[(op.op, op.axis)] = result.get((op.op, op.axis), 1) * op.amt
  return result

def normalize_suggestion(default_opt, suggestion):
  return tuple([Opt(op, axis, amt) for (op, axis), amt in ({(op, axis): amt for op, axis, amt in default_opt} | suggestion_to_dict(suggestion)).items()])

def kernel_optimize_search(pool, k:Linearizer, create_k:Callable[[], Linearizer], baseline, bufs, var_vals, suggestion):
  import nevergrad as ng
  opts, default_opt = kernel_optimize_opts(k, suggestion)
  if not opts: return "BASELINE"
  search_space = prod([len(x.choices) for x in opts])
  st = time.perf_counter()
  budget = getenv("BUDGET", 2000)
  num_workers = pool._processes
  optimizer = ng.optimizers.NGOpt(parametrization=ng.p.Tuple(*opts), budget=min(search_space, budget), num_workers=num_workers)
  #print(k.full_shape, [y for y in normalize_suggestion(default_opt, suggestion) if y.amt != 1])
  optimizer.suggest(normalize_suggestion(default_opt, suggestion))
  optimizer.register_callback("tell", (bar := ng.callbacks.ProgressBar()))

  from extra.helpers import _CloudpickleFunctionWrapper
  best, best_ran, best_name = 10_000, 0, ""
  q: List[Any] = []
  ran = 0
  while optimizer.num_tell < cast(int, optimizer.budget):
    while len(q) < num_workers and optimizer.num_ask < cast(int, optimizer.budget):
      ask = optimizer.ask()
      try:
        this_k = create_k()
        apply_ng_opt(this_k, ask.value)
      except Exception:
        if DEBUG >= 3: traceback.print_exc()
        optimizer.tell(ask, 10_000, constraint_violation=1.0)
        q.append((None, None))
      else:
        q.append((ask, start_compile(pool, this_k)))
    while len(q) > num_workers-1 or (optimizer.num_ask == optimizer.budget and q):
      ask, prg = q.pop(0)
      if prg is None: continue
      try:
        name, prg = prg.get(timeout=5)
        tm = run_and_time(prg, bufs, var_vals, baseline=baseline)
      except Exception:
        if DEBUG >= 3: traceback.print_exc()
        optimizer.tell(ask, 10_000, constraint_violation=1.0)
      else:
        optimizer.tell(ask, tm)
        ran += 1
        if tm < best: best, best_ran, best_name = tm, ran, name
        bar._progress_bar.set_description(f"{baseline:7.3f}/{best:7.3f} ({baseline / best * 100:4.0f}%) @ {best_ran:4}/{ran:4} - {best_name + ' ' * (37 - ansilen(best_name))}")
  recommendation = optimizer.provide_recommendation()
  et = time.perf_counter() - st
  del bar, optimizer
  if DEBUG >= 1: print(f"optimizer({et:6.2f} s to search) space {search_space:8d} with tm {recommendation.loss:5.2f} ms vs baseline {baseline:5.2f} ms, a {baseline/recommendation.loss:5.2f}x gain : {k.colored_shape()}")
  return apply_ng_opt(create_k(), recommendation.value) if recommendation.loss < baseline else "BASELINE"

def fix_opt_axes(create_k, opts):
  k = create_k()
  axis_idxs = list(range(len(k.full_shape)))
  fixed_ops = []
  for opt in opts:
    axis = opt.axis
    if axis >= k.first_reduce: axis -= k.local_dims + len(k.group_for_reduce)
    fixed_ops.append(Opt(opt.op, axis_idxs[axis], opt.amt))

    if opt.amt == k.full_shape[axis]: axis_idxs.pop(axis)

    k.apply_opt(opt)

  return fixed_ops

def apply_ng_opt(k, x):
  axis_idxs = list(range(len(k.full_shape)))
  orig_first_reduce = k.first_reduce
  for op, axis, amt in x:
    if amt == 1: continue

    pre_axis = axis = axis_idxs.index(axis)
    if axis >= orig_first_reduce: axis += k.local_dims + len(k.group_for_reduce)

    if amt == k.full_shape[axis]: axis_idxs.pop(pre_axis)

    k.apply_opt(Opt(op, axis, amt))

  assert prod(k.full_shape[k.shape_len - k.upcasted:k.shape_len]) <= 512, "too many upcasts!"
  assert prod(k.full_shape[k.first_reduce - k.local_dims:k.first_reduce + len(k.group_for_reduce)]) <= 1024, "too many locals!"

  return k.applied_opts
