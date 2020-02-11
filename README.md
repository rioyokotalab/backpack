# A fork of [BackPACK](https://github.com/f-dangel/backpack)

## Installation
```bash
git clone git@github.com:rioyokotalab/backpack.git
cd ~/backpack
pip install -e .
```

## Jacobian computation by BackPACK vs for-loop
```bash
$ python test/benchmark_jacobian.py
bs: 32
input_ndim: 1000
n_hidden: 3
hidden_ndim: 1000
output_ndim: 10
trial: 100
----------------
device cuda
jacobian with loop: 2.750s
jacobian with backpack: 0.695s
```
