
import os
import json
import pandas as pd
import numpy as np
from glob import glob

from rsub import *
from matplotlib import pyplot as plt

data = {}
for f in glob('_results/AS/*log.jl'):
    vals = pd.DataFrame([json.loads(x) for x in open(f)])
    vals = vals.groupby('run_id').value.max().values
    data[os.path.basename(f)] = vals

res = [(k, np.max(v)) for k,v in data.items()]

df = pd.DataFrame(res)
df['graph'] = df[0].apply(lambda x: x.split('.')[0])
df['decay'] = df[0].apply(lambda x: x.split('-')[1])
df['value'] = df[1]
del df[0]
del df[1]

df.graph = df.graph.apply(lambda x: int(x[1:]))

df = df.sort_values(['graph', 'decay'])

df[df.decay == 'decay']
