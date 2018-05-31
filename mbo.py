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
    parser.add_argument('--inpath', type=str, default='_data/wiki/wiki.txt')
    parser.add_argument('--outpath', type=str, default='_results/wiki/wiki')
    
    parser.add_argument('--tau', type=float, default=20)           # Time horizon for diffusion
    parser.add_argument('--inner-iters', type=int, default=100)    # Number of steps in inner loop
    parser.add_argument('--outer-iters', type=int, default=25)     # Number of steps in outer loop
    parser.add_argument('--conv-iters', type=int, default=2)       # Number of outer loops to look back for convergence
    parser.add_argument('--conv-thresh', type=float, default=0.01) # break if not improved by `conv_thresh` after `conv_iters` outer
    
    parser.add_argument('--n-jobs', type=int, default=32) # Number of processors
    parser.add_argument('--n-runs', type=int, default=32) # Number of times to run MBO
    
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--plot', action='store_true')
    return parser.parse_args()


def mbo(u, adj, L, tau=20, inner_iters=100, outer_iters=25, conv_iters=2, conv_thresh=0.01):
    best_value = -1
    
    conv_lookback = inner_iters * conv_iters
    
    cuts = [{
        "outer_iter" : 0,
        "inner_iter" : -1,
        "value"      : adj[u == 1][:,u != 1].sum(),
    }]
    
    for outer_iter in range(outer_iters):
        
        # Signless diffusion
        for inner_iter in range(inner_iters):
            u -= (tau / inner_iters) * L.dot(u)
            
            # Value of cut
            value = adj[u > 0][:,u <= 0].sum()
            cuts.append({
                "outer_iter" : outer_iter,
                "inner_iter" : inner_iter,
                "value"      : value,
            })
            
            if value > best_value:
                best_u = u.copy()
        
        # Threshold step
        u = (2.0 * (u > 0) - 1)
        
        if len(cuts) > conv_lookback:
            prev_value = cuts[-conv_lookback]['value']
            if value < prev_value * (1 + conv_thresh):
                break
        
    return cuts, best_value, best_u

# --
# Run

if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    
    # --
    # IO
    
    print('mbo.py: loading %s' % args.inpath, file=sys.stderr)
    edges = pd.read_csv(args.inpath, sep='\t', header=None).values
    
    # --
    # Prep
    
    num_nodes = len(set(np.hstack(edges)))
    num_edges = edges.shape[0]
    print('num_nodes=%d | num_edges=%d' % (num_nodes, num_edges), file=sys.stderr)
    
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
    
    # --
    # Run
    
    def runner(seed, **kwargs):
        np.random.seed(seed)
        u = np.random.choice((-1.0, 1.0), adj.shape[0])
        return mbo(u=u, **kwargs)
    
    t = time()
    mbo_args = {
        "adj"         : adj,
        "L"           : L,
        "tau"         : args.tau,
        "inner_iters" : args.inner_iters,
        "outer_iters" : args.outer_iters,
        "conv_iters"  : args.conv_iters,
        "conv_thresh" : args.conv_thresh,
    }
    jobs = [delayed(runner)(seed=seed, **mbo_args) for seed in range(args.n_runs)]
    results = Parallel(n_jobs=args.n_jobs)(jobs)
    
    results, best_values, best_us = zip(*results)
    
    best_u = best_us[np.argmax(best_values)]
    np.save(args.outpath + '-u', best_u)
    
    # --
    # Log
    
    best_vals = [max([r['value'] for r in result]) for result in results]
    print(json.dumps({
        "max_val"      : np.max(best_vals),
        "mean_val"     : np.mean(best_vals),
        "min_val"      : np.min(best_vals),
        "elapsed_time" : time() - t,
    }), file=sys.stderr)
    
    outfile = open(args.outpath + '-log.jl', 'w')
    for run_id, result in enumerate(results):
        for r in result:
            print(json.dumps({
                "run_id"     : run_id,
                "outer_iter" : r['outer_iter'],
                "inner_iter" : r['inner_iter'],
                "value"      : r['value'],
            }), file=outfile)
    
    outfile.close()
    
    # --
    # Plot
    
    if args.plot:
        for run_id, result in enumerate(results):
            progress = [r['outer_iter'] + r['inner_iter'] / args.inner_iters for r in result]
            values   = [r['value'] for r in result]
            _ = plt.plot(progress, values)
        
        _ = plt.title('MBO+ (tau=%f | inner_iters=%d | outer_iters=%d)' % (args.tau, args.inner_iters, args.outer_iters))
        _ = plt.xlabel('outer_iter')
        _ = plt.ylabel('value')
        plt.savefig(args.outpath + '-plot.png')
