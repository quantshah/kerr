# kerr
Kerr resonator simulations. Please note that the code is in development and has not been tested. The relevant article for the theory can be found in:

Bartolo, Nicola, et al. "Exact steady state of a Kerr resonator with one-and two-photon driving and dissipation: Controllable Wigner-function multimodality and dissipative phase transitions." Physical Review A 94.3 (2016): 033841.

# Installation

- clone the git repository using

```
git clone https://github.com/sahmed95/kerr.git
```

- Install using the `develop` command so that changes in the source are reflected immediately

```
python setup.py develop
```

# Use

```
from kerr.steady import Kerr

f = 0.5j
g = 0.5j
c = 0.5j

resonator = Kerr(f, g, c)
norm = resonator.normalization()
corr = resonator.correlation(3, 4)
W = resonator.wigner(2j+5)

print(norm, corr, W)
```
