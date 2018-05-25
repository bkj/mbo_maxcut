#### mbo_maxcut

Approximate MAXCUT solver using the [MBO+](https://arxiv.org/pdf/1711.02419.pdf) (Euler method).

#### Installation

```
conda create -n mbo_env python=2.7 pip -y
source activate mbo_env
pip install -r requirements.txt
```

#### Usage

See `./run.sh` for usage.

#### Todo

- Parallel sparse-dense matrix-vector multiplication
- CUDA implementation (via `cupy`)
- Implement other graph Laplacians (right now, only have random walk (`delta_1`) which is most performant)