import ctypes
import time

import numpy as np

from tinygrad import dtypes
from tinygrad.device import Device, Buffer
from tinygrad.helpers import getenv, DEBUG

if Device.DEFAULT == "AMD":
  from tinygrad.runtime.ops_amd import AMDDevice, CACHE_FLUSH_AND_INV_TS_EVENT, hsa, HWPM4Queue

#devices = [Device[f"{Device.DEFAULT}:{i}"] for i in range(getenv("GPUS", 6))]
devices = [Device[f"{Device.DEFAULT}:{i}"] for i in [2, 4, 3, 1, 5, 0]]

# test latency of consecutive timestamps
def ts_test(d, q_t):
  rounds = 100
  signals = [d._get_signal() for _ in range(rounds)]
  d.synchronize()
  q = q_t()
  for i in range(rounds):
    q.timestamp(signals[i])
  q.signal(d.timeline_signal, d.timeline_value)
  d.timeline_value += 1
  q.submit(d)
  d._wait_signal(d.timeline_signal, d.timeline_value-1)

  t = [d._read_timestamp(sig) for sig in signals]
  lat = [b - a for a, b in zip(t, t[1:])]
  if DEBUG >= 1: print('latencies', sum(lat) / len(lat), lat)
  medlat = sorted(lat)[(rounds-1) // 2]
  if DEBUG >= 1: print('median latency', medlat)

  d.signals_pool.extend(signals)
  return medlat



def ping_test(q1_t, q2_t, d1, d2):
  if DEBUG >= 1: print('timesync', d1.dname, d2.dname)
  warmup = 100
  rounds = 100
  d1.synchronize()
  d2.synchronize()
  signals1, signals2 = [d1._get_signal() for _ in range(warmup + rounds + 1)], [d2._get_signal() for _ in range(warmup + rounds)]

  q1, q2 = q1_t(), q2_t()
  if getenv("READ_REMOTE_SIG"):
    # timestamp on self device, signal self device, wait on other device
    q1.timestamp(signals1[0]).signal(signals1[0], 1)
    for i in range(warmup + rounds):
      q2.wait(signals1[i], 1).timestamp(signals2[i]).signal(signals2[i], 1)
      q1.wait(signals2[i], 1).timestamp(signals1[i + 1]).signal(signals1[i + 1], 1)
  else:
    # timestamp on self device, signal other device, wait on self device
    q1.timestamp(signals1[0]).signal(signals2[0], 1)
    for i in range(warmup + rounds):
      q2.wait(signals2[i], 1).timestamp(signals2[i]).signal(signals1[i+1], 1)
      q1.wait(signals1[i+1], 1).timestamp(signals1[i + 1])
      if i + 1 < len(signals2):
        q1.signal(signals2[i + 1], 1)

  q1.signal(d1.timeline_signal, d1.timeline_value)
  d1.timeline_value += 1
  if hasattr(q2, 'bind'): q2.bind(d2)
  if hasattr(q1, 'bind'): q1.bind(d1)
  st = time.perf_counter_ns()
  q2.submit(d2)
  q1.submit(d1)
  d1._wait_signal(d1.timeline_signal, d1.timeline_value - 1)
  et = time.perf_counter_ns()

  t1 = [d1._read_timestamp(sig) for sig in signals1[warmup:]]
  t2 = [d2._read_timestamp(sig) for sig in signals2[warmup:]]

  diff1 = [b - a for a, b in zip(t1, t2)]
  diff2 = [a - b for a, b in zip(t1[1:], t2)]
  ping = [(a + b) // 2 for a, b in zip(diff1, diff2)]
  offsets = [(a - b) // 2 for a, b in zip(diff1, diff2)]
  if DEBUG >= 1: print('ping', sum(ping) / len(ping), ping)
  if DEBUG >= 1: print('offests', sum(offsets) / len(offsets), offsets)
  medping = sorted(ping)[rounds // 2]
  offset = sorted(offsets)[rounds // 2]
  d1.signals_pool.extend(signals1)
  d2.signals_pool.extend(signals2)
  return offset, medping


def print_matrix(matrix, width=5):
  print(' ' * 6 + '[' + '] ['.join(' '.join(f"{devices[j].dname:>{width}}" for l in range(2)) for j in range(len(devices))) + ']')
  for i in range(len(devices)):
    for k in range(2):
      print(f'{devices[i].dname:>5} ' + '[' + '] ['.join(' '.join(f"{matrix[i][j][k][l] or 0:>{width}}" for l in range(2)) for j in range(len(devices))) + ']')


def run_pings():
  roffm = [[[[0, 0], [0, 0]] for _ in range(len(devices))] for _ in range(len(devices))]
  rpingm = [[[[0, 0], [0, 0]] for _ in range(len(devices))] for _ in range(len(devices))]
  for i in range(len(devices)):
    for j in range(len(devices)):
      dev1, dev2 = devices[i], devices[j]
      for k, q1 in enumerate([dev1.hw_compute_queue_t, dev1.hw_copy_queue_t]):
        for l, q2 in enumerate([dev2.hw_compute_queue_t, dev2.hw_copy_queue_t]):
          if i == j and q1 is q2: continue
          if DEBUG >= 1: print(dev1.dname, dev2.dname, q1, q2)
          offset, ping = ping_test(q1, q2, dev1, dev2)
          roffm[i][j][k][l] = offset
          rpingm[i][j][k][l] = ping
  return roffm, rpingm

def driver_ts_lat():
  compute_ts_lat = ts_test(devices[0], devices[0].hw_compute_queue_t)
  copy_ts_lat = ts_test(devices[0], devices[0].hw_copy_queue_t)
  print('compute_ts_lat', compute_ts_lat)
  print('copy_ts_lat', copy_ts_lat)

def driver_clock_sync():
  offset_matrix, ping_matrix = run_pings()

  print('ping matrix:')
  print_matrix(ping_matrix)

  print('sleep 10s')
  time.sleep(10)

  offset_matrix2, _ = run_pings()

  print('sleep 100s')
  time.sleep(90)

  offset_matrix3, _ = run_pings()

  diff_matrix1 = ((np.array(offset_matrix2, dtype=np.int64) - np.array(offset_matrix, dtype=np.int64)) * 10)
  diff_matrix2 = (np.array(offset_matrix3, dtype=np.int64) - np.array(offset_matrix, dtype=np.int64))
  print_matrix((diff_matrix1 - diff_matrix2).tolist(), width=10)
  print_matrix((diff_matrix2 / diff_matrix1).tolist(), width=10)


def sdma_copy_test(d1, d2):
  b1, b2 = Buffer(d1.dname, 2 ** 30, dtype=dtypes.uint8).allocate(), Buffer(d2.dname, 2 ** 30, dtype=dtypes.uint8).allocate()
  d1._gpu_map(b2._buf)
  q_t = d1.hw_copy_queue_t
  q = q_t()
  signals = [d1._get_signal() for _ in range(2)]
  q.timestamp(signals[0]).copy(b2._buf.va_addr, b1._buf.va_addr, b1.nbytes).timestamp(signals[1])
  q.signal(d1.timeline_signal, d1.timeline_value).signal(d1.timeline_signal, d1.timeline_value - 1)
  d1.timeline_value += 1

  for _ in range(10):
    q.submit(d1)
    d1._wait_signal(d1.timeline_signal, d1.timeline_value - 1)
    ts = [d1._read_timestamp(signal) for signal in signals]
    print(ts[1] - ts[0])
  d1.signals_pool.extend(signals)


sdma_copy_test(devices[0], devices[0])