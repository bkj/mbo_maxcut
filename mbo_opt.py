#!/usr/bin/env python

"""
    mbo.py
"""

from __future__ import division, print_function

import sys
import json
import argparse
import numpy as np
import pandas as pd
from time import time
from scipy import sparse

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from joblib import Parallel, delayed

# --

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str, default='_data/set1/g54.rud')
    parser.add_argument('--outpath', type=str, default='delete-me')
    parser.add_argument('--hotstart', type=str)
    
    parser.add_argument('--conv-iters', type=int, default=5)        # Number of outer loops to look back for convergence
    parser.add_argument('--conv-thresh', type=float, default=0.0)   # break if not improved by `conv_thresh` after `conv_iters` outer
    
    parser.add_argument('--n-jobs', type=int, default=32) # Number of processors
    parser.add_argument('--n-runs', type=int, default=32) # Number of times to run MBO
    
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--plot', action='store_true')
    return parser.parse_args()


def mbo(u, adj, L, lr, inner_iters, outer_iters, conv_iters, conv_thresh, decay, verbose=False):
    
    if decay:
        def scheduler(progress):
            return lr * float(outer_iters - progress) / outer_iters
    else:
        def scheduler(progress):
            return lr
    
    best_value = -1
    
    cuts = [{
        "outer_iter" : 0,
        "value"      : adj[u > 0].dot(u <= 0).sum()
    }]
    
    t = time()
    for outer_iter in range(outer_iters):
        
        # Signless diffusion
        for inner_iter in range(inner_iters):
            progress = outer_iter + (inner_iter / inner_iters)
            u -= scheduler(progress) * L.dot(u)
            
        # Value of cut
        value = adj[u > 0].dot(u <= 0).sum()
        cuts.append({
            "outer_iter"   : outer_iter,
            "value"        : value,
            "elapsed_time" : time() - t,
        })
        if verbose:
            print(json.dumps(cuts[-1]))
        
        if value > best_value:
            best_u = u.copy()
        
        # Threshold step
        u = (2.0 * (u > 0) - 1)
        
        if len(cuts) > conv_iters:
            prev_value = cuts[-conv_iters]['value']
            if value < prev_value * (1 + conv_thresh):
                break
    
    return cuts, best_value, best_u


# --
# Run

import dlib
def dlib_find_max_global(f, bounds, **kwargs):
    varnames = f.__code__.co_varnames[:f.__code__.co_argcount]
    bound1_, bound2_ = [], []
    for varname in varnames:
        bound1_.append(bounds[varname][0])
        bound2_.append(bounds[varname][1])
    
    return dlib.find_max_global(f, bound1_, bound2_, **kwargs)

if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    
    # --
    # IO
    
    print('mbo.py: loading %s' % args.inpath, file=sys.stderr)
    # >>
    edges = pd.read_csv(args.inpath, sep='\t', header=None).values
    # edges = pd.read_csv(args.inpath, sep=' ', header=None, skiprows=1)
    # del edges[2]
    # edges = edges.values
    # <<
    
    # --
    # Prep
    
    num_nodes = len(set(np.hstack(edges)))
    num_edges = edges.shape[0]
    print('mbo.py: num_nodes=%d | num_edges=%d' % (num_nodes, num_edges), file=sys.stderr)
    
    adj_dim = edges.max() + 1
    adj = sparse.csr_matrix((np.ones(edges.shape[0]), (edges[:,0], edges[:,1])), shape=(adj_dim, adj_dim))
    adj = ((adj + adj.T) > 0).astype(np.float)
    
    # Drop nodes w/ degree 0
    sel = adj.getnnz(axis=0) > 0
    adj = adj[sel][:,sel]
    
    # Make signless random walk Laplacia
    d = np.asarray(adj.sum(axis=-1)).squeeze()
    L = adj + sparse.diags(d)
    L = sparse.diags(1 / d).dot(L)
    
    def runner(lr, inner_iters, outer_iters, decay)
        print('mbo.py: running', file=sys.stderr)
        
        inits = {
            "discrete" : lambda: np.random.choice((-1.0, 1.0), adj.shape[0]),
            "uniform"  : lambda: np.random.uniform(-1, 1, adj.shape[0]),
            "normal"   : lambda: np.random.normal(0, 1, adj.shape[0]),
        }
        
        def runner(seed, **kwargs):
            np.random.seed(seed)
            u = inits[kwargs['init']]()
            del kwargs[args.init]
            return mbo(u=u, **kwargs)
        
        t = time()
        mbo_args = {
            "adj"         : adj,
            "L"           : L,
            
            "lr"          : lr,
            "inner_iters" : inner_iters,
            "outer_iters" : outer_iters,
            
            "conv_iters"  : args.conv_iters,
            "conv_thresh" : args.conv_thresh,
            "decay"       : True
        }
        
        jobs = [delayed(runner)(seed=args.seed + seed, **mbo_args) for seed in range(args.n_runs)]
        results = Parallel(n_jobs=args.n_jobs)(jobs)
        
        results, best_values, best_us = zip(*results)
        best_value = max([max([r['value'] for r in result]) for result in results])
        print(best_value)
        return best_value
    
    best_args, best_score = dlib_find_max_global(run_one, bounds={
        "lr"          : (0, 1),
        "inner_iters" : (32, 512),
        "outer_iters" : (8, 128),
    }, num_function_calls=100, solver_epsilon=0.001)
