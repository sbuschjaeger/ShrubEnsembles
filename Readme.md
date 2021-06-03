# PRoxIMale Ensembles -- PRIME

Ensemble Learning via (biased) proximal gradient descent implemented in Python and C++. 


# Extend this code

Activate the conda environment

    conda env create -f environment.yml
    conda activate prime
    pip install -e .

Have some fun with the Python code.

# Compile

Compiler is installed inside the enviroment. The setup script uses
`-DCMAKE_CXX_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX`if they are set. If not, we let CMake try to figure out whats going on. This only happens on windows. 

# Acknowledgements 

The software is written and maintained by [Sebastian Buschjäger](https://sbuschjaeger.github.io/) as part of his work at the [Chair for Artificial Intelligence](https://www-ai.cs.tu-dortmund.de) at the TU Dortmund University and the [Collaborative Research Center 876](https://sfb876.tu-dortmund.de). If you have any question feel free to contact me under sebastian.buschjaeger@tu-dortmund.de 

Special thanks goes to Sibylle Hess (s.c.hess@tue.nl) for taking care of math outside my comfort zone.