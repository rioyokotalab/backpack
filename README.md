# BACKpropagation PACKage - a backpack for `PyTorch`

| branch | tests & examples | coverage |
|--------|---------------------------|----------|
|`master` | [![Build Status](https://travis-ci.org/f-dangel/backpack.svg?branch=master)](https://travis-ci.org/f-dangel/backpack) | [![Coverage Status](https://coveralls.io/repos/github/f-dangel/backpack/badge.svg?branch=master)](https://coveralls.io/github/f-dangel/backpack) |
| `development` | [![Build Status](https://travis-ci.org/f-dangel/backpack.svg?branch=development)](https://travis-ci.org/f-dangel/backpack) | [![Coverage Status](https://coveralls.io/repos/github/f-dangel/backpack/badge.svg?branch=development)](https://coveralls.io/github/f-dangel/backpack) |

A backpack for PyTorch that extends the backward pass of feedforward networks to compute quantities beyond the gradient.

- Check out the [cheatsheet](examples/cheatsheet.pdf) for an overview of quantities.
- Check out the [examples](https://f-dangel.github.io/backpack/) on how to use the code.

## How to cite
If you are using `backpack` for your research, consider citing the [paper](https://openreview.net/forum?id=BJlrF24twB) 
```
@inproceedings{dangel2020backpack,
    title     = {Back{PACK}: Packing more into Backprop},
    author    = {Felix Dangel and Frederik Kunstner and Philipp Hennig},
    booktitle = {International Conference on Learning Representations},
    year      = {2020},
    url       = {https://openreview.net/forum?id=BJlrF24twB}
}
```
