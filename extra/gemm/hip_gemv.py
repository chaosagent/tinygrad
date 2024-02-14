import time
import numpy as np
from tinygrad import dtypes, Tensor
from tinygrad.helpers import getenv, prod, flat_mv, Context, DEBUG
from tinygrad.runtime.ops_hip import HIPAllocator, HIPDevice, HIPProgram, compile_hip
from tinygrad.renderer.cstyle import HIPLanguage
from tinygrad.device import CompiledASTRunner, Device, Buffer
from tinygrad import TinyJit

# AMD_LOG_LEVEL=3 ./MIOpenDriver gemm --iter 1000 --time 1 --a_w 2048 --a_h 2048 --b_w 2048
# 5.5: Cijk_Ailk_Bljk_HHS_BH_MT128x128x16_MI16x16x16x1_SN_1LDSB0_APM1_ABV0_ACED0_AF0EM1_AF1EM1_AMAS3_ASE_ASGT_ASAE01_ASCE01_ASEM1_AAC0_BL1_BS1_DTL0_DTVA0_DVO0_ETSP_EPS1_FL0_GRVW8_GSU1_GSUASB_GLS0_ISA1100_IU1_K1_KLA_LBSPP128_LPA0_LPB8_LDL1_LRVW16_LWPMn1_LDW0_FMA_MIAV1_MDA2_NTA0_NTB0_NTC0_NTD0_NEPBS0_NLCA1_NLCB1_ONLL1_OPLV0_PK0_PAP0_PGR1_PLR1_RK0_SIA1_SS1_SU32_SUM0_SUS128_SCIUI1_SPO0_SRVW0_SSO0_SVW4_SNLL0_TT4_64_TLDS1_USFGROn1_VAW2_VSn1_VW4_WSGRA1_WSGRB1_WS32_WG32_4_1_WGM4
# 5.6: Cijk_Ailk_Bljk_HHS_BH_MT128x128x16_MI16x16x16x1_SN_1LDSB0_APM1_ABV0_ACED0_AF0EM1_AF1EM1_AMAS3_ASE_ASGT_ASLT_ASAE01_ASCE01_ASEM1_AAC0_BL1_BS1_DTL0_DTVA0_DVO0_ETSP_EPS1_FL0_GRPM1_GRVW8_GSU1_GSUASB_GLS0_ISA1100_IU1_K1_KLA_LBSPP128_LPA0_LPB8_LDL1_LRVW16_LWPMn1_LDW0_FMA_MIAV1_MDA2_MO40_NTA0_NTB0_NTC0_NTD0_NEPBS0_NLCA1_NLCB1_ONLL1_OPLV0_PK0_PAP0_PGR1_PLR1_RK0_SIA1_SS1_SU32_SUM0_SUS128_SCIUI1_SPO0_SRVW0_SSO0_SVW4_SNLL0_TT4_64_TLDS1_USFGROn1_VAW2_VSn1_VW4_WSGRA1_WSGRB1_WS32_WG32_4_1_WGM4
# gets ~100
# hipExtModuleLaunchKernel ( 0x0x16ccde0, 2048, 16, 1, 128, 1, 1,
# 161.60 us = 106.31 TFLOPS
# with --batch_count 8 / 1.258128 ms / (8*2048*2048*2048*2)/(1.258128)*1e-9 / 109.24 TFLOPS

# we only get ~53
# KY=2 KX=2 N=2048 python3 extra/gemm/hip_matmul.py
#   4194304    324.76 us, would be  52899.88 GFLOPS matmul, 154.98 GB/s

N = getenv("N", 24576)
KX = getenv("KX", 4)
LX = getenv("LX", 1)
MBLOCK = getenv("MBLOCK", 32)
KBLOCK = getenv("KBLOCK", 12)
assert N%(KBLOCK*KX) == 0, f"N must be multiple of {KBLOCK*KX}"
FLOPS = N*N*2  # mul and acc
BW = N*N*2 + N*4 + N*N*2//(MBLOCK*KX*LX)  # include vector and output bw
MBW = N*N*2  # only matrix bw

# Can HIPAllocator initialized as device=0 by default?
device = Device["HIP"]
hipallocator = device.allocator
a = Buffer(device.dname, N, dtypes.float32)
b = Buffer(device.dname, N*N, dtypes.float16)
c = Buffer(device.dname, N, dtypes.float16)
na = np.empty(N, np.float32)
nb = np.random.default_rng().standard_normal(size=(N,N), dtype=np.float32).astype(np.float16)
nc = np.random.default_rng().standard_normal(size=(N,1), dtype=np.float32).astype(np.float16)
hipallocator.copyin(b._buf, bytearray(nb))
hipallocator.copyin(c._buf, bytearray(nc))

src = (f"""
#define F32
typedef float float8 __attribute__((ext_vector_type(8)));
typedef _Float16 half{KBLOCK} __attribute__((ext_vector_type({KBLOCK})));
{HIPLanguage.kernel_prefix}
void __launch_bounds__ ({MBLOCK * LX}) test(float* c, half* a, half* b) {{
  const int gx = {HIPLanguage.code_for_workitem['g'](0)} * {LX} + {HIPLanguage.code_for_workitem['l'](1)};

  const int lIdx = {HIPLanguage.code_for_workitem['l'](0)};
  const int lane = lIdx;

  c += gx*{KX*MBLOCK} + lane*{KX};
  a += gx*{KX*MBLOCK};

  half{KBLOCK} a_frag[{KX}], a_frag1[{KX}];
  half{KBLOCK} b_frag, b_frag1;
  float c_frag[{KX}] = {{}};
  for (int ele = 0; ele < {KBLOCK}; ++ele) {{
    for (int x = 0; x < {KX}; x++) {{
      a_frag1[x][ele] = a[(0+ele)*{N} + x + lane*{KX}];
    }}
  }}
  for (int ele = 0; ele < {KBLOCK}; ++ele) {{
    b_frag1[ele] = b[0+ele];
  }}

  for (int k = 0; k < {N}; k += {KBLOCK}) {{
    {HIPLanguage.barrier}
    
    // swap regs
    for (int ele = 0; ele < {KBLOCK}; ++ele) {{
      for (int x = 0; x < {KX}; x++) {{
        a_frag[x][ele] = a_frag1[x][ele];
      }}
    }}
    for (int ele = 0; ele < {KBLOCK}; ++ele) {{
      b_frag[ele] = b_frag1[ele];
    }}
    
    if (k+{KBLOCK} < {N}) {{
      // load next regs
      for (int ele = 0; ele < {KBLOCK}; ++ele) {{
        for (int x = 0; x < {KX}; x++) {{
          a_frag1[x][ele] = a[(k+ele+{KBLOCK})*{N} + x + lane*{KX}];
        }}
      }}
      for (int ele = 0; ele < {KBLOCK}; ++ele) {{
        b_frag1[ele] = b[k+ele+{KBLOCK}];
      }}
    }}
    
    // mul
    for (int ele = 0; ele < {KBLOCK}; ele++) {{
      for (int x = 0; x < {KX}; x++) {{
        c_frag[x] = ((float) a_frag[x][ele]) * ((float) b_frag[ele]) + c_frag[x];
      }}
    }}
  }}
  
  for (int x = 0; x < {KX}; x++) {{
    c[x] = c_frag[x];
  }}
}}""")
lib = compile_hip(src)

global_size, local_size = [N//(KX*MBLOCK*LX), 1, 1], [MBLOCK, LX, 1]
print("global/local size", global_size, local_size, f"local_size:{prod(local_size)} total_size:{prod(global_size+local_size)}")
binary = HIPProgram(device.device, "test", lib)
prog = CompiledASTRunner(None, "test", src, device, global_size=global_size, local_size=local_size, precompiled=lib)

if DEBUG >= 6: exit(0)
scratch = Tensor.rand(1024, 1024, 128, device=device.dname).realize()  # 7900 xtx l3 cache is 96mb

def timeit(fxn):
  scratch.assign(-scratch).realize()
  st = time.perf_counter()
  et = fxn()
  ret = time.perf_counter() - st # NOTE: et doesn't contain the launch overhead
  #print(f"{ret*1e6:.2f} us")
  return et

JITCNT=8
@TinyJit
def test_fn():
  for _ in range(JITCNT):
    prog.exec([a, b, c])
test_fn(); test_fn();
assert len(test_fn.jit_cache) == 1
#tm = min([timeit(lambda: test_fn.jit_cache[0].prg([], {}, wait=True)/JITCNT) for _ in range(1000)])
tm = min([timeit(lambda: prog([a, b, c], {}, wait=True)) for _ in range(1000)])
hipallocator.copyout(flat_mv(na.data),a._buf)
na = na.reshape(N,1)
comp = nb.astype(np.float32).transpose(1,0) @ nc.astype(np.float32)
print(f"{N*N:10d} {tm*1e6:9.2f} us, would be {FLOPS*1e-9/tm:9.2f} GFLOPS matmul, {BW*1e-9/tm:.2f} GB/s, {MBW*1e-9/tm:.2f} matrix GB/s")
np.testing.assert_allclose(na, comp, atol=1e-2, rtol=1e-2)
