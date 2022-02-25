# Shrub Ensembles

Shrub Ensembles are ensembles of small decision trees (= shrubs) that a refined via (proximal) SGD. This is a fairly optimized C++ implementation with Python binding. I will add some documentation in the future. 

This code has been developed for our paper

> Shrub Ensembles for Online Classification ([arxiv-preprint](https://arxiv.org/abs/2112.03723)) by Buschjäger, Sebastian, Hess, Sibylle, and Morik, Katharina in Proceedings of the Thirty-Sixth AAAI Conference on Artificial Intelligence (AAAI-22) 2022

For the exact experiments performed in this paper check out https://github.com/sbuschjaeger/se-online

# Use this code in Python

First install the python package

    pip install git+https://github.com/sbuschjaeger/ShrubEnsembles.git

Then you can use the classes `{MASE, GASE, OSE}` directly. You can find an example in `tests/debug.py` 

# Use this code in C++

Simply clone this repository

    git clone git@github.com:sbuschjaeger/ShrubEnsembles.git

and add the header files in `src/se` to your project. Expect for the Python binding there are no additional `cpp` files. This code is header-only. It is tested with various gcc versions that support C++17. I don't know if C++11/14 would also work. An example on how to use the code is given in `tests/debug.cpp`. Also check out the `CMakeLists.txt` for compilation-related problems.

# Acknowledgements 

The software is written and maintained by [Sebastian Buschjäger](https://sbuschjaeger.github.io/) as part of his work at the [Chair for Artificial Intelligence](https://www-ai.cs.tu-dortmund.de) at the TU Dortmund University and the [Collaborative Research Center 876](https://sfb876.tu-dortmund.de). If you have any question feel free to contact me under sebastian.buschjaeger@tu-dortmund.de 

Special thanks goes to Sibylle Hess (s.c.hess@tue.nl) for taking care of math outside my comfort zone.