
import numpy as np
import numpy.random as rand
import json
import sys

count = int(sys.argv[2])
   
def randomSelect(choices):
    return choices[rand.randint(0, len(choices))]

def randomRange(low, high, step, incl_upper=True):
    if incl_upper:
        lst = list(np.arange(low, high+step, step))
    else:
        lst = list(np.arange(low, high, step))
    return randomSelect(lst)   

def randomIntRange(low, high, step, incl_upper=True):
    return int(np.round(randomRange(low, high, step, incl_upper)))

def make_level():
    d = {}
    d['samplingFraction'] = randomRange(0.001, 1.0, 0.001)#0.025 + 0.975 * rand.rand()
    d['optimizer'] = randomSelect(['sgd'])
    d['downsamplingFactor'] = randomSelect([1, 2, 3, 4])
    d['smoothingSigma'] = randomRange(0.0, 7.0, 0.01)
    d['alphaLevels'] = randomIntRange(1, 63, 1)
    d['normalization'] = randomRange(0.0, 0.05, 0.0025)
    d['gradientMagnitude'] = randomSelect(['false'])
    d['learningRate'] = randomRange(0.25, 3.0, 0.25)
    d['lambdaFactor'] = randomRange(0.01, 0.3, 0.01)
    d['samplingMode'] = randomSelect(['quasi', 'uniform'])
    d['seed'] = randomIntRange(1, 10000, 1)
    di = {}
    di['iterations'] = randomIntRange(100, 3000, 100)
    di['controlPoints'] = randomIntRange(7, 72, 1)
    d['innerParams'] = [di]
    return d


rand.seed(1337)


for it in range(1, count+1):
    levels = randomIntRange(1, 5, 1)
    paramSets = []
    for k in range(levels):
        paramSets.append(make_level())
    dall = {'paramSets': paramSets}

    filename = '%s/params%d.json' % (sys.argv[1], it)
    with open(filename, 'w') as outfile:
        json.dump(dall, outfile)

