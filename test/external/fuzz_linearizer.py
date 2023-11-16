import random, traceback
import numpy as np
from collections import Counter
from extra.optimization.helpers import load_worlds, ast_str_to_lin
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.features.search import get_linearizer_actions, bufs_from_lin, tuplize_uops
from tinygrad.graph import print_tree
from tinygrad.helpers import ImageDType, prod, getenv, DEBUG
from tinygrad.ops import Device, Compiled, Interpreted
from tinygrad.ops import Device, Compiled, Interpreted, get_interpreted_fxn
from tinygrad.lazy import vars_from_ast

device = Device[Device.DEFAULT]


def run_linearizer(lin: Linearizer, rawbufs=None, var_vals=None):
  if rawbufs is None: rawbufs = bufs_from_lin(lin)
  if var_vals is None: var_vals = {v: v.min for v in vars_from_ast(lin.ast)}

  # TODO: images needs required_optimization
  try:
    if isinstance(device, Compiled):
      prg = device.to_program(lin)
    else:
      prg = get_interpreted_fxn(device.fxn_for_op, device.from_underlying, lin.ast)
  except:
    print(lin.ast)
    traceback.print_exc()
    print("COMPILE FAILED!!")
    return "COMPILE_ERROR"

  try:
    prg.exec(rawbufs, var_vals, force_wait=True)
  except:
    print(lin.ast)
    traceback.print_exc()
    print("EXEC FAILED!!")
    return "EXEC_ERROR"

  return "PASS"


def fuzz_linearizer(lin: Linearizer, seed=42):
  og_lin = lin.copy()
  base_opts = len(og_lin.applied_opts)
  random.seed(seed)
  np.random.seed(seed)
  if DEBUG >= 1: print_tree(lin.ast)
  if DEBUG >= 1: print(lin.colored_shape())
  rawbufs = bufs_from_lin(lin)

  seen_uops = {}
  iters = 0
  reroll_in_row = 0
  ground_truth = None
  def reroll():
    opts = random.choice(list(seen_uops.values()))
    lin = og_lin.copy()
    for opt in opts[base_opts:]: lin.apply_opt(opt)
    return lin
  while 1:
    if len(seen_uops) >= 200 or iters >= 400: break  # enough for this kernel
    iters += 1
    if ground_truth is not None:
      actions = get_linearizer_actions(lin, include_0=False)
      if not actions:
        if reroll_in_row >= 5: break
        reroll_in_row+=1
        lin = reroll()
        continue
      lin = random.choice(list(actions.values()))
      if lin.applied_opts and DEBUG >= 1: print(f"applied action: {lin.applied_opts[-1]}")

    # stop if kernel uops repeat
    tuops = tuplize_uops(lin.linearize().uops)
    if tuops in seen_uops:
      if reroll_in_row >= 5: break
      reroll_in_row += 1
      lin = reroll()
      continue
    seen_uops[tuops] = tuple(lin.applied_opts)
    reroll_in_row = 0

    if DEBUG >= 0: print(lin.colored_shape())
    # get a new output buffer
    rawbufs[0] = type(rawbufs[0])(rawbufs[0].size, rawbufs[0].dtype)
    var_vals = {v: random.randint(v.min, v.max) for v in vars_from_ast(lin.ast)}
    if (msg := run_linearizer(lin, rawbufs, var_vals)) != "PASS":
      print(f"{lin.applied_opts=}")
      return msg

    result = rawbufs[0].toCPU()
    if ground_truth is None:
      ground_truth = result
    else:
      try:
        np.testing.assert_allclose(result, ground_truth, rtol=1e-2, atol=1e-2)
      except AssertionError:
        print(lin.ast)
        traceback.print_exc()
        print(f"{lin.applied_opts=}")
        return "NOT_ALLCLOSE"
  return "PASS"


if __name__ == "__main__":
  ast_strs = load_worlds()
  print(f"{len(ast_strs)=}")
  tested = 0
  c = Counter()
  failed = []
  for i, ast in enumerate(ast_strs[:getenv("FUZZ_N", len(ast_strs))]):
    if "Variable" in ast and isinstance(device, Interpreted): continue  # no symbolic shape for Interpreted
    if "dtypes.image" in ast and Device.DEFAULT != "GPU": continue  # IMAGE is only for GPU
    tested += 1
    ast = ast.replace('dtypes.float', 'dtypes.half')
    lin = ast_str_to_lin(ast)
    used_tc = lin.apply_tensor_cores(hand_coded=False)
    print(f"testing ast {i}")
    if not used_tc: continue
    base_opts = len(lin.applied_opts)
    seen_uops = {}
    fuzz = str(fuzz_linearizer(lin))
    c[fuzz] += 1
    print(fuzz)
    if fuzz != "PASS":
      failed.append(i)
  print(f"{tested=}")
  print(c.most_common())
  print(f"{failed=}")