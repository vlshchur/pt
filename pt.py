#!/usr/bin/env python3

#BUILDING:
#python3 setup.py build_ext --inplace

import argparse
import sys
import time
from planar_tree import planar_tree
from random import randrange
import numpy as np

parser = argparse.ArgumentParser(description='Migration inference from PSMC.')

parser.add_argument('tree_size', type=int,
                    help='tree size')
parser.add_argument('--seed', '-seed', nargs=1, type=int, default=None,
                    help='random seed')


clargs = parser.parse_args()

if isinstance(clargs.seed, list):
    clargs.seed = clargs.seed[0]

if clargs.seed == None:
    rndseed = randrange(sys.maxsize)
else:
    rndseed = clargs.seed
print("Seed: ", rndseed)

pt = planar_tree(clargs.tree_size)
pt.GenerateRandomTree()
t1 = time.time()
nodeId = pt.CalculateCladeSizes()
t2 = time.time()
if nodeId != 0:
    pt.GetSequences(nodeId)
t3 = time.time()
if nodeId != 0:
    pt.WriteSequences("test.txt")
t4 = time.time()
print(t2 - t1)
print(t3 - t2)
print(t4 - t3)
print("_________________________________")
