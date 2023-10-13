from typing import List
from tqdm import tqdm
import multiprocessing as mp
from models.resnet import ResNet50
from tinygrad.tensor import Tensor
from tinygrad.ops import LoadOps, Device, Compiled
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.codegen.search import bufs_from_lin, time_linearizer, get_linearizer_actions, Opt, OptOps
from tinygrad.helpers import ansilen, DEBUG, getenv, prod
from tinygrad.graph import print_tree
from tinygrad.lazy import vars_from_ast
from tinygrad.shape.symbolic import sym_infer
from extra.optimization.kopt_ng import run_and_time, fix_opt_axes
from extra.optimization.helpers import compile_kernel, start_compile, catch_exception

import shelve
global_db = shelve.open("./greedy_cache")

def apply_wino_upcast(self):
  return
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
    self.apply_opt(Opt(OptOps.UPCAST, axis, upcast_amt))
  return self

if __name__ == "__main__":
  mdl = ResNet50()
  seen = set()

  pool = mp.Pool(getenv("WORKERS", 32))

  # the device we are optimizing for
  device: Compiled = Device[Device.DEFAULT]
  print(f"optimizing for {Device.DEFAULT}")

  # first model run to init the weights, they are saved in seen
  mdl(Tensor.empty(64, 3, 224, 224)).lazydata.schedule(seen)

  # run model again to get only what changes, these are the kernels of the model
  x = Tensor.empty(64, 3, 224, 224)
  out = mdl(x)
  sched = out.lazydata.schedule(seen)
  sched = [x for x in sched if x.ast.op not in LoadOps]

  # work with the schedule
  total_tm = 0
  running_gflops = 0
  for i,si in enumerate(sched):
    if DEBUG >= 2: print_tree(si.ast)

    # create output/input buffers (NOTE: bufs_from_lin is slower, so we don't use it. TODO: fix)
    rawbufs = [device.buffer(si.out.st.size(), si.out.dtype)] + [device.buffer(x.st.size(), x.dtype) for x in si.inputs]
    #rawbufs = bufs_from_lin(lin)
    var_vals = {k: k.min for k in vars_from_ast(si.ast)}

    # "linearize" the op into uops in different ways
    lins:List[Linearizer] = []

    # always try hand coded opt
    lin = Linearizer(si.ast, device.linearizer_opts)
    lin.hand_coded_optimizations()
    lins.append(lin)
    baseline = run_and_time(compile_kernel(lin, checks=False)[1], rawbufs, var_vals)

    # maybe try tensor cores
    lin = Linearizer(si.ast, device.linearizer_opts)
    if lin.apply_tensor_cores():
      lins.append(lin)

    # try a greedy search
    if getenv("GREEDY"):
      lin = Linearizer(si.ast, device.linearizer_opts)
      if str(lin.ast) in global_db:
        for ao in global_db[str(lin.ast)]:
          lin.apply_opt(ao)
        best_time = run_and_time(compile_kernel(lin)[1], rawbufs, var_vals)
      else:
        apply_wino_upcast(lin)
        best, best_time = lin, 10_000
        for greedy_i in range(15):
          acted_lins = get_linearizer_actions(lin)
          compiling = [(i, start_compile(pool, lin)) for i, lin in acted_lins.items()]
          timed_lins = {i: catch_exception(run_and_time, on_fail=float('inf'))(prg.get(timeout=5)[1], rawbufs, var_vals) for i, prg in tqdm(compiling, desc=f'greedy layer {greedy_i}')}
          opts = sorted(timed_lins.items(), key=lambda x: x[1])
          if opts[0][0] == 0: break   # we are done
          lin = acted_lins[opts[0][0]]
          if opts[0][1] < best_time:
            best = lin
            best_time = opts[0][1]
          if DEBUG >= 0: print(f"{opts[0][1]:10.2f} ms from {len(opts):3d} actions", lin.colored_shape())
        lin = best
        global_db[str(lin.ast)] = lin.applied_opts
      lins.append(lin)
      baseline = min(baseline, best_time)
    if getenv("KOPT"):
      if str(('KOPT', si.ast)) in global_db:
        ng_choice = global_db[str(('KOPT', si.ast))]
      else:
        from extra.optimization.kopt_ng import kernel_optimize_search
        lin = Linearizer(si.ast, device.linearizer_opts)
        apply_wino_upcast(lin)
        create_k = lambda: apply_wino_upcast(Linearizer(si.ast, device.linearizer_opts))
        ng_choice = kernel_optimize_search(pool, lin, create_k, baseline, rawbufs, var_vals, fix_opt_axes(create_k, lins[-1].applied_opts))
        global_db[str(('KOPT', si.ast))] = lin.applied_opts
      if ng_choice != "BASELINE":
        for op in ng_choice: lin.apply_opt(op)
        lins.append(lin)

    # benchmark the programs
    choices = []
    for lin in lins:
      tm = time_linearizer(lin, rawbufs, allow_test_size=False, cnt=10, should_copy=False)
      gflops = sym_infer(lin.info.flops, {k:k.min for k in vars_from_ast(lin.ast)})*1e-9/tm
      choices.append((tm, gflops, lin))

      # print all kernels
      if DEBUG >= 1: print(f"                 kernel {i:2d} {lin.display_name+' '*(37-ansilen(lin.display_name))} {str(lin.global_size):18s} {str(lin.local_size):12s} takes {tm*1000:7.2f} ms, {gflops:6.0f} GFLOPS")
    tm, gflops, lin = sorted(choices, key=lambda x: x[0])[0]
    print(f"*** {total_tm*1000:7.2f} ms : kernel {i:2d} {lin.display_name+' '*(37-ansilen(lin.display_name))} {str(lin.global_size):18s} {str(lin.local_size):12s} takes {tm*1000:7.2f} ms, {gflops:6.0f} GFLOPS")
    total_tm += tm
    running_gflops += gflops * tm
  print(f"******* total {total_tm*1000:.2f} ms, {running_gflops/total_tm:6.0f} GFLOPS")
