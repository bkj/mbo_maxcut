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
    
    parser.add_argument('--lr', type=float, default=0.2)            # Time horizon for diffusion
    parser.add_argument('--no-decay', action="store_true")          # Linear decay on learning rate
    parser.add_argument('--init', type=str, default='normal')       # How to initialize solution
    
    parser.add_argument('--inner-iters', type=int, default=100)     # Number of steps in inner loop
    parser.add_argument('--outer-iters', type=int, default=25)      # Number of steps in outer loop
    parser.add_argument('--conv-iters', type=int, default=5)        # Number of outer loops to look back for convergence
    parser.add_argument('--conv-thresh', type=float, default=0.0)   # break if not improved by `conv_thresh` after `conv_iters` outer
    
    parser.add_argument('--n-jobs', type=int, default=32) # Number of processors
    parser.add_argument('--n-runs', type=int, default=32) # Number of times to run MBO
    
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--plot', action='store_true')
    return parser.parse_args()

# import localsolver
# def run_local(edges, hotstart):
#     with localsolver.LocalSolver() as ls:
#         sel = edges[:,0] > edges[:,1]
#         edges[sel] = edges[sel][:,[1, 0]]
        
#         origin, dest = zip(*set(map(tuple, edges)))
        
#         unodes  = np.unique(np.hstack([np.unique(origin), np.unique(dest)]))
        
#         lookup = dict(zip(unodes, range(len(unodes))))
#         origin = [lookup[x] for x in origin]
#         dest   = [lookup[x] for x in dest]
        
#         num_nodes = len(unodes)
#         num_edges = edges.shape[0]
        
#         print('defining problem', file=sys.stderr)
#         x          = [ls.model.bool() for i in range(num_nodes)]
#         incut      = [ls.model.neq(x[o], x[d]) for o, d in zip(origin, dest)]
#         cut_weight = ls.model.sum(incut)
        
#         ls.model.maximize(cut_weight)
#         ls.model.close()
        
#         hotstart = (hotstart == 1).astype(int)
#         _ = [xx.set_value(h) for xx, h in zip(x, hotstart)]
        
#         ls.param.nb_threads = 32
#         ls.create_phase().time_limit = 1
        
#         print('running', file=sys.stderr)
#         ls.solve()
        
#         solution = np.array([xx.value for xx in x]).astype(np.float64)
#         return solution

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
    
    # --
    # Run
    print('mbo.py: running', file=sys.stderr)
    
    inits = {
        "discrete" : lambda: np.random.choice((-1.0, 1.0), adj.shape[0]),
        "uniform"  : lambda: np.random.uniform(-1, 1, adj.shape[0]),
        "normal"   : lambda: np.random.normal(0, 1, adj.shape[0]),
    }
    
    def runner(seed, **kwargs):
        np.random.seed(seed)
        u = inits[args.init]()
        return mbo(u=u, **kwargs)
    
    t = time()
    mbo_args = {
        "adj"         : adj,
        "L"           : L,
        "lr"          : args.lr,
        "inner_iters" : args.inner_iters,
        "outer_iters" : args.outer_iters,
        "conv_iters"  : args.conv_iters,
        "conv_thresh" : args.conv_thresh,
        "decay"       : not args.no_decay,
    }
    
    # jobs = []
    # for seed in range(args.n_runs):
    #     np.random.seed(seed)
    #     u = inits[args.init]()
    #     u = run_local(edges, u)
    #     jobs.append(delayed(runner)(u, **mbo_args))
    
    jobs = [delayed(runner)(seed=args.seed + seed, **mbo_args) for seed in range(args.n_runs)]
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
                "value"      : r['value'],
            }), file=outfile)
    
    outfile.close()
    
    # --
    # Plot
    
    if args.plot:
        for run_id, result in enumerate(results):
            progress = [r['outer_iter'] for r in result]
            values   = [r['value'] for r in result]
            _ = plt.plot(progress, values)
        
        _ = plt.title('MBO+ (lr=%f | inner_iters=%d | outer_iters=%d)' % (args.lr, args.inner_iters, args.outer_iters))
        _ = plt.xlabel('outer_iter')
        _ = plt.ylabel('value')
        plt.savefig(args.outpath + '-plot.png')
