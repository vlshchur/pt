# cython: language_level=3
# cython: initializedcheck = False
# distutils: language = c++

cimport cython

from libc.math cimport log, floor, abs
from libcpp.vector cimport vector
from mc_lib.rndm cimport RndmWrapper

import numpy as np
import sys

include "fast_choose.pxi"


def print_err(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


# use named constants for event types
#DEF BIRTH = 0

cdef class Node:
    cdef:
        double time
        Py_ssize_t type_, population, haplotype, newHaplotype, newPopulation

    def __init__(self, double time, Py_ssize_t type_, Py_ssize_t population, Py_ssize_t haplotype, Py_ssize_t newHaplotype, Py_ssize_t newPopulation):
        self.time = time
        self.type_ = type_
        self.population = population
        self.haplotype = haplotype
        self.newHaplotype = newHaplotype
        self.newPopulation = newPopulation

cdef class Nodes:
    cdef:
        double[::1] times
        Py_ssize_t size
        #Py_ssize_t[::1] mutations
        Py_ssize_t[:,::1] clade, mutations

    def __init__(self, Py_ssize_t size_):
        self.size = size_
        self.times = np.zeros(self.size, dtype=float)
        #self.child = np.zeros( (self.size, 2), dtype=bint)#list of decendants if internal, garbage if leaf
        self.clade = np.zeros((self.size,2), dtype=int)#number of decendant leaves
        self.mutations = np.zeros((self.size,2), dtype=int)#single mutation per node: position and allele - FIXME

    cdef void CladeSize(self, Py_ssize_t nodeId, Py_ssize_t clade_left, clade_right):
        self.clade[nodeId,0] = clade_left
        self.clade[nodeId,1] = clade_right

    cdef (Py_ssize_t, Py_ssize_t) GetClade(self, Py_ssize_t nodeId):
        return self.clade[nodeId,0], self.clade[nodeId,1]

    cdef (Py_ssize_t, Py_ssize_t) GetMutation(self, Py_ssize_t nodeId):
        return self.mutations[nodeId,0], self.mutations[nodeId,1]

    cdef void AddMutation(self, Py_ssize_t nodeId, Py_ssize_t pos, Py_ssize_t allele):
        self.mutations[nodeId, 0] = pos
        self.mutations[nodeId, 1] = allele

cdef class planar_tree:
    cdef:

        RndmWrapper rndm
        Py_ssize_t leaf_num, seq_len, rootId, path_ptr, seq_num
        Py_ssize_t[::1] leaves, path
        Py_ssize_t[:,::1] sequences
        double[::1] times
        Nodes nodes

        #double currentTime, rn, totalRate, maxEffectiveBirth, totalMigrationRate
        #Py_ssize_t bCounter, dCounter, sCounter, migCounter, mutCounter, popNum, dim, hapNum, susceptible_num, migPlus, migNonPlus, swapLockdown
        #Events events
        #PopulationModel pm
        #Mutations mut

        #int[::1] tree, suscType
        #int[:,::1] liveBranches

        #double[::1] bRate, dRate, sRate, tmRate, migPopRate, popRate, times, pm_maxEffectiveMigration, maxSusceptibility, elementsArr2, immunePopRate, infectPopRate, sourceSuscepTransition, suscepCumulTransition
        #double[:,::1] pm_migrationRates, pm_effectiveMigration, birthHapPopRate, tEventHapPopRate, hapPopRate, mRate, susceptibility, totalHapMutType, suscepTransition, immuneSourcePopRate
        #double[:,:,::1] eventHapPopRate, susceptHapPopRate, hapMutType

    def __init__(self, leaf_num, rndseed=2020):
        self.rndm = RndmWrapper(seed=(rndseed, 0))
        self.leaf_num = leaf_num
        self.seq_len = 30000
        self.leaves = np.zeros(self.leaf_num, dtype=int)
        self.path = np.zeros(self.leaf_num, dtype=int)
        for i in range(self.leaf_num):
            self.leaves[i] = i
        self.times  = np.zeros(self.leaf_num, dtype=float)

        self.nodes = Nodes(2*self.leaf_num)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef (Py_ssize_t, Py_ssize_t) CladeSize(self, Py_ssize_t nodeId):
        cdef:
            double nodeTime
            Py_ssize_t nodeLeft, nodeRight
        nodeTime = self.times[nodeId]
        nodeLeft = nodeId - 1
        while self.times[nodeLeft] <= nodeTime:
            nodeLeft -= 1
        nodeRight = nodeId
        while nodeRight < self.leaf_num-1 and self.times[nodeRight+1] <= nodeTime:
            nodeRight += 1
        return nodeLeft, nodeRight

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void GetRoot(self):
        cdef:
            double rt
        rt = 0.0
        for i in range(1, self.leaf_num):
            if self.times[i] > rt:
                self.rootId = i
                rt = self.times[i]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void GenerateRandomTree(self, double maxtime=20.0):
        cdef:
            double rnd
        self.times[0] = maxtime + 1.0
        for i in range(1, self.leaf_num):
            rnd = self.rndm.uniform()
            self.times[i] = rnd*maxtime
        self.GetRoot()
        self.GenerateRandomMutations()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void GenerateRandomMutations(self):
        cdef:
            Py_ssize_t pos, allele
        for i in range(0, self.leaf_num-1):
            pos = int ( floor(self.rndm.uniform()*self.seq_len) )
            allele = int( 1+floor(self.rndm.uniform()*3) )
            self.nodes.AddMutation(self.leaf_num+i, pos, allele)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void GetSequences(self, Py_ssize_t clade_id):
        cdef:
            Py_ssize_t cl, cr, pos, allele, cl_i, cr_i
        cl, cr = self.nodes.GetClade(self.leaf_num+clade_id)
        self.seq_num = cr - cl + 1
        self.sequences = np.zeros((self.seq_len, self.seq_num), dtype=int)

        self.GetRootPath(clade_id)
        for i in range(self.path_ptr):
            nodeId = self.path[i]
            pos, allele = self.nodes.GetMutation(self.leaf_num+nodeId)
            cl_i, cr_i = self.nodes.GetClade(self.leaf_num+nodeId)
            #print(nodeId, pos, allele, cl_i, cr_i)
            for j in range(cl_i, cr_i+1):
                self.sequences[pos, j] = allele

        for i in range(cl+1, cr+1):
            pos, allele = self.nodes.GetMutation(self.leaf_num+i)
            cl_i, cr_i = self.nodes.GetClade(self.leaf_num+i)
            for j in range(cl_i, cr_i+1):
                self.sequences[pos, j] = allele

    def WriteSequences(self, fn):
        f_seq = open(fn, 'w')
        for i in range(self.seq_len):
            if sum(self.sequences[i]) > 0:
                for j in range(self.seq_num):
                    f_seq.write(str(self.sequences[i, j]))
            f_seq.write("\n")
        f_seq.close()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void GetRootPath(self, Py_ssize_t nodeId):
        cdef:
            Py_ssize_t cl, cr, clade_id, clade_id_old
        self.path_ptr = 0
        clade_id_old = 0
        clade_id = nodeId
        while 1:
            if clade_id == clade_id_old:
                break
            clade_id_old = clade_id
            cl, cr = self.nodes.GetClade(self.leaf_num+clade_id)
            if cl == 0:
                if self.times[cr+1] >= self.times[self.rootId]:
                    break
                else:
                    clade_id = cr+1
                    self.path[self.path_ptr] = clade_id
                    self.path_ptr += 1
            elif cr == self.leaf_num - 1:
              if self.times[cl] >= self.times[self.rootId]:
                  break
              else:
                  clade_id = cl
                  self.path[self.path_ptr] = clade_id
                  self.path_ptr += 1
            else:
                clade_id = cl if self.times[cl] < self.times[cr+1] else cr+1
                if clade_id >= self.leaf_num:
                    print(clade_id)
                    return
                self.path[self.path_ptr] = clade_id
                self.path_ptr += 1
        print()


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef Py_ssize_t CalculateCladeSizes(self):
        cdef:
            Py_ssize_t cl, cr, nodeId
            int out_counter
        out_counter = 1
        nodeId = 0
        for i in range(0, self.leaf_num):
            cl, rl = self.CladeSize(i)
            self.nodes.CladeSize(self.leaf_num+i, cl, rl)
            if rl-cl+1 > 2000 and rl-cl+1 < 2100 and out_counter > 0:
                print("i cl rl size", i, cl, rl, rl-cl+1)
                nodeId = i
                out_counter -= 1
        return nodeId

    def PrintTree(self):
        for i in range(self.leaf_num):
            print(self.times[i], " ", end="")
        print()

    def PrintTreeRange(self, left, right):
        for i in range(left, right):
            print(i, " ", self.times[i])
        print()
