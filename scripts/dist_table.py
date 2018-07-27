
# Script for extracting the distribution of registration errors

import numpy as np
import csv
import sys

def load_file(path):
    with open(path, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        return [list_to_numbers(row) for row in reader if len(row) > 1]

def list_to_numbers(Xs):
    #print(Xs)
    return np.sort(np.asarray(Xs, dtype=np.float64))

# Create the distribution of values at -3sigma, ..., 3sigma
def make_distribution(A):
    A = np.sort(A)
    p = np.asarray(np.round((A.size-1) * np.asarray([0.0, 1.0 - 0.9973, 1.0 - 0.9545, 1.0-0.6827, 0.5, 0.6827, 0.9545, 0.9973, 1.0])), dtype = np.int64)
    return A[p]

def make_table_row(A, row_title, is_max):
    s = "" + row_title

    for i in xrange(len(A)):
        if is_max[i]:
            s = s + (' & \\textbf{%.3f}' % (A[i]))
        else:
            s = s + (' & %.3f' % (A[i]))

    return s + " \\\\"

#B = np.arange(10000)
#b = make_distribution(B)

#print(make_table_row(b, "SSD"))

if len(sys.argv) > 1:
    rows = load_file(sys.argv[1])
    names = ["SSD", "NCC", "MI", "$\\alpha_{\\text{ASMD}}$", "$\\alpha_{\\text{SMD}}$"]

    row_count = len(rows)

    dist_list = [make_distribution(r) for r in rows]
    dist_min = dist_list[0]

    for i in xrange(row_count):
        dist_min = np.minimum(dist_min, dist_list[i])

    for i in xrange(len(rows)):
        print make_table_row(dist_list[i], names[i], dist_list[i] <= dist_min)
