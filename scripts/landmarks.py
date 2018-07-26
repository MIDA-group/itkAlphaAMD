
import csv
import numpy as np         

def read_csv(path, skip_first = False):
    with open(path, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        if skip_first:
            reader.next()
        arr = []
        for row in reader:
            arr.append(np.asarray(row, dtype=np.float))
        return np.asarray(arr, dtype=np.float)

def euclidean(lm1, lm2):
	return np.sqrt(np.sum(np.square(lm1-lm2), axis=1))

def mean_euclidean(lm1, lm2):
	return np.mean(euclidean(lm1, lm2))

def distribution(d, percentiles):
	ds = np.sort(d)
	indices = np.int(percentiles * d.shape[0])
	return ds[indices]

# Helper function for smd

def min_distance(p, set):
    mind = float('inf')
    mini = -1
    for i in xrange(set.shape[0]):
        setp = set[i, :]
        d = np.sqrt(np.sum(np.square(p-setp)))
        if d < mind:
            mind = d
            mini = i
    return (mind, mini)

# Sum of minimal distance between reference and target set of landmarks

def smd(ref, target):
    refshape = ref.shape
    acc = 0
    for i in xrange(refshape[0]):
        (d, ind) = min_distance(ref[i, :], target)
        acc = acc + d
    return acc

if __name__ == "__main__":
  ts1 = [[1, 2, 3], [4, 5, 6]]
  ts2 = [[2, 2, 3], [0, 0, 0]]

  print(euclidean(np.asarray(ts1), np.asarray(ts2)))
  print(mean_euclidean(np.asarray(ts1), np.asarray(ts2)))
  print(smd(np.asarray(ts1), np.asarray(ts2)))

