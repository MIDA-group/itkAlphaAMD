
import errno
import os
import sys

import multiprocessing
import subprocess
import shlex

import timeit

from multiprocessing.pool import ThreadPool

import numpy as np
import csv

def param_dict_to_string(param_dict):
  s = ""
  for key, value in param_dict.iteritems():
    s = s + " -" + key + " " + value
  return s


def makedir_string(path):
  return "mkdir " + path


def check_exists(path, name):
  if not os.path.isfile(path):
    raise Exception(name + " file \'" + path + "\' does not exist.")


def create_directory(path):
  try:
    os.makedirs(path)
  except OSError as exc:  # Python >2.5
    if exc.errno == errno.EEXIST and os.path.isdir(path):
      pass
    else:
      raise


def cpu_count():
  return multiprocessing.cpu_count()


def create_pool(cpu_count):
  if cpu_count <= 0:
    return (ThreadPool(multiprocessing.cpu_count()), [])
  else:
    return (ThreadPool(cpu_count), [])


def join_pool(pool):
  pool[0].close()
  pool[0].join()


def run_process(cmd, ind, output_path=None):
  if output_path == None:
      # subprocess.call(shlex.split(cmd))  # This will block until cmd finishes
      start = timeit.default_timer()
      p = subprocess.Popen(shlex.split(
          cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      out, err = p.communicate()
      stop = timeit.default_timer()
      if not ind == None:
        sys.stdout.write("[%d:%f]" % (ind, stop-start))
        sys.stdout.flush()
      return (out, err)
  else:
      stdout_path=output_path + "out.txt"
      stderr_path=output_path + "err.txt"
      with open(stdout_path, "wb") as outf, open(stderr_path, "wb") as errf:
          # subprocess.call(shlex.split(cmd))  # This will block until cmd finishes
          start=timeit.default_timer()
          # stdout=subprocess.PIPE, stderr=subprocess.PIPE)
          p=subprocess.Popen(shlex.split(cmd), stdout=outf, stderr=errf)
          out, err=p.communicate()
          stop=timeit.default_timer()
          if not ind == None:
            sys.stdout.write("[%d:%f]" % (ind, stop-start))
            sys.stdout.flush()
          return (out, err)

def run(pool, cmd, output_path=None):
  ind=len(pool[1])
  pool[1].append(pool[0].apply_async(run_process, (cmd, ind, output_path,)))
  return ind

def get_result(pool, index):
  return pool[1][index]

def read_csv(path, skip_first = False):
    with open(path, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        if skip_first:
            reader.next()
        arr = []
        for row in reader:
            arr.append(np.asarray(row, dtype=np.float))
        return np.asarray(arr, dtype=np.float)
# for result in results:
#    out, err = result.get()
#    print("out: {} err: {}".format(out, err))
# subprocess.call("./merge_resized_images")
