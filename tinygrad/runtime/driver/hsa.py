import ctypes
import tinygrad.runtime.autogen.hsa as hsa
from tinygrad.helpers import init_c_var

def check(status):
  if status != 0: raise RuntimeError(f"HSA Error {status}")

# Precalulated AQL info
AQL_PACKET_SIZE = ctypes.sizeof(hsa.hsa_kernel_dispatch_packet_t)
EMPTY_SIGNAL = hsa.hsa_signal_t()

DISPATCH_KERNEL_SETUP = 3 << hsa.HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS
DISPATCH_KERNEL_HEADER  = 1 << hsa.HSA_PACKET_HEADER_BARRIER
DISPATCH_KERNEL_HEADER |= hsa.HSA_FENCE_SCOPE_SYSTEM << hsa.HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE
DISPATCH_KERNEL_HEADER |= hsa.HSA_FENCE_SCOPE_SYSTEM << hsa.HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE
DISPATCH_KERNEL_HEADER |= hsa.HSA_PACKET_TYPE_KERNEL_DISPATCH << hsa.HSA_PACKET_HEADER_TYPE

BARRIER_HEADER  = 1 << hsa.HSA_PACKET_HEADER_BARRIER
BARRIER_HEADER |= hsa.HSA_FENCE_SCOPE_SYSTEM << hsa.HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE
BARRIER_HEADER |= hsa.HSA_FENCE_SCOPE_SYSTEM << hsa.HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE
BARRIER_HEADER |= hsa.HSA_PACKET_TYPE_BARRIER_AND << hsa.HSA_PACKET_HEADER_TYPE

class HWQueue:
  def __init__(self, device, sz=-1):
    self.device = device
    self.wait_signals = []

    check(hsa.hsa_agent_get_info(self.device.agent, hsa.HSA_AGENT_INFO_QUEUE_MAX_SIZE, ctypes.byref(max_queue_size := ctypes.c_uint32())))
    queue_size = min(max_queue_size.value, sz) if sz != -1 else max_queue_size.value

    null_func = ctypes.CFUNCTYPE(None, hsa.hsa_status_t, ctypes.POINTER(hsa.struct_hsa_queue_s), ctypes.POINTER(None))()
    self.hw_queue = init_c_var(ctypes.POINTER(hsa.hsa_queue_t)(), lambda x: check(
      hsa.hsa_queue_create(self.device.agent, queue_size, hsa.HSA_QUEUE_TYPE_SINGLE, null_func, None, (1<<32)-1, (1<<32)-1, ctypes.byref(x))))

    self.write_addr = self.hw_queue.contents.base_address
    self.write_addr_end = self.hw_queue.contents.base_address + (64 * self.hw_queue.contents.size) - 1
    self.next_doorbell_index = -1

    check(hsa.hsa_amd_profiling_set_profiler_enabled(self.hw_queue, 1))

  def __del__(self):
    if hasattr(self, 'hw_queue'): check(hsa.hsa_queue_destroy(self.hw_queue))

  def submit_kernel(self, prg, global_size, local_size, kernargs, need_signal=False):
    signal = self.device.alloc_signal(returnable=True) if need_signal else EMPTY_SIGNAL

    packet = hsa.hsa_kernel_dispatch_packet_t.from_address(self.write_addr)
    packet.workgroup_size_x = local_size[0]
    packet.workgroup_size_y = local_size[1]
    packet.workgroup_size_z = local_size[2]
    packet.reserved0 = 0
    packet.grid_size_x = global_size[0] * local_size[0]
    packet.grid_size_y = global_size[1] * local_size[1]
    packet.grid_size_z = global_size[2] * local_size[2]
    packet.private_segment_size = prg.private_segment_size
    packet.group_segment_size = prg.group_segment_size
    packet.kernel_object = prg.handle
    packet.kernarg_address = kernargs
    packet.reserved2 = 0
    packet.completion_signal = signal
    packet.setup = DISPATCH_KERNEL_SETUP
    packet.header = DISPATCH_KERNEL_HEADER

    self.next_doorbell_index += 1
    self.write_addr += AQL_PACKET_SIZE
    if self.write_addr > self.write_addr_end: self.write_addr = self.hw_queue.contents.base_address

    hsa.hsa_queue_store_write_index_relaxed(self.hw_queue, self.next_doorbell_index + 1)
    hsa.hsa_signal_store_screlease(self.hw_queue.contents.doorbell_signal, self.next_doorbell_index)
    return signal

  def submit_barrier(self, wait_signals=None, need_signal=False):
    assert wait_signals is None or len(wait_signals) < 5
    signal = self.device.alloc_signal(returnable=True) if need_signal else EMPTY_SIGNAL

    packet = hsa.hsa_barrier_and_packet_t.from_address(self.write_addr)
    packet.reserved0 = 0
    packet.reserved1 = 0
    packet.dep_signal[0] = wait_signals.pop() if wait_signals and len(wait_signals) > 0 else EMPTY_SIGNAL
    packet.dep_signal[1] = wait_signals.pop() if wait_signals and len(wait_signals) > 0 else EMPTY_SIGNAL
    packet.dep_signal[2] = wait_signals.pop() if wait_signals and len(wait_signals) > 0 else EMPTY_SIGNAL
    packet.dep_signal[3] = wait_signals.pop() if wait_signals and len(wait_signals) > 0 else EMPTY_SIGNAL
    packet.dep_signal[4] = wait_signals.pop() if wait_signals and len(wait_signals) > 0 else EMPTY_SIGNAL
    packet.reserved2 = 0
    packet.completion_signal = signal
    packet.header = BARRIER_HEADER

    self.next_doorbell_index += 1
    self.write_addr += AQL_PACKET_SIZE
    if self.write_addr > self.write_addr_end: self.write_addr = self.hw_queue.contents.base_address

    hsa.hsa_queue_store_write_index_relaxed(self.hw_queue, self.next_doorbell_index + 1)
    hsa.hsa_signal_store_screlease(self.hw_queue.contents.doorbell_signal, self.next_doorbell_index)
    return signal

  def wait(self):
    signal = self.submit_barrier(need_signal=True)
    hsa.hsa_signal_wait_scacquire(signal, hsa.HSA_SIGNAL_CONDITION_LT, 1, (1 << 64) - 1, hsa.HSA_WAIT_STATE_ACTIVE)

def find_agent(typ, device_id):
  @ctypes.CFUNCTYPE(hsa.hsa_status_t, hsa.hsa_agent_t, ctypes.c_void_p)
  def __filter_agents(agent, data):
    status = hsa.hsa_agent_get_info(agent, hsa.HSA_AGENT_INFO_DEVICE, ctypes.byref(device_type := hsa.hsa_device_type_t()))
    if status == 0 and device_type.value == typ:
      ret = ctypes.cast(data, ctypes.POINTER(hsa.hsa_agent_t))
      if ret[0].handle == device_id:
        ret[0] = agent
        return hsa.HSA_STATUS_INFO_BREAK
      ret[0].handle = ret[0].handle + 1
    return hsa.HSA_STATUS_SUCCESS

  hsa.hsa_iterate_agents(__filter_agents, ctypes.byref(agent := hsa.hsa_agent_t()))
  return agent

def find_memory_pool(agent, segtyp=-1, flags=-1, location=-1):
  @ctypes.CFUNCTYPE(hsa.hsa_status_t, hsa.hsa_amd_memory_pool_t, ctypes.c_void_p)
  def __filter_amd_memory_pools(mem_pool, data):
    check(hsa.hsa_amd_memory_pool_get_info(mem_pool, hsa.HSA_AMD_MEMORY_POOL_INFO_SEGMENT, ctypes.byref(segment := hsa.hsa_amd_segment_t())))
    if segtyp >= 0 and segment.value != segtyp: return hsa.HSA_STATUS_SUCCESS

    check(hsa.hsa_amd_memory_pool_get_info(mem_pool, hsa.HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, ctypes.byref(fgs := hsa.hsa_amd_memory_pool_global_flag_t())))
    if flags >= 0 and (fgs.value & flags) == flags: return hsa.HSA_STATUS_SUCCESS

    check(hsa.hsa_amd_memory_pool_get_info(mem_pool, hsa.HSA_AMD_MEMORY_POOL_INFO_LOCATION, ctypes.byref(loc := hsa.hsa_amd_memory_pool_location_t())))
    if location >= 0 and loc.value != location: return hsa.HSA_STATUS_SUCCESS

    check(hsa.hsa_amd_memory_pool_get_info(mem_pool, hsa.HSA_AMD_MEMORY_POOL_INFO_SIZE, ctypes.byref(sz := ctypes.c_size_t())))
    if sz == 0: return hsa.HSA_STATUS_SUCCESS

    ret = ctypes.cast(data, ctypes.POINTER(hsa.hsa_amd_memory_pool_t))
    ret[0] = mem_pool
    return hsa.HSA_STATUS_INFO_BREAK

  hsa.hsa_amd_agent_iterate_memory_pools(agent, __filter_amd_memory_pools, ctypes.byref(region := hsa.hsa_amd_memory_pool_t()))
  return region
