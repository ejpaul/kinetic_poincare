# kinetic_poincare

This repository contains example scripts to be run with the orbit_resonance branch of simsopt (https://github.com/hiddenSymmetries/simsopt/tree/orbit_resonance) and booz_xform (https://github.com/hiddenSymmetries/booz_xform). The equilibrium corresponds with the vacuum quasi-helical configuration from "Magnetic Fields with Precise Quasisymmetry for Plasma Confinement", scaled to ARIES-CS size and field strength. 

The directories ending in _perfect correspond with those for which all of the symmetry-breaking harmonics are artificially suppressed. This enables a cleaner calculation of the characteristic frequencies, which are saved in the files omega_perfect.txt and rad_perfect.txt. This files are then read in by the ``imperfect'' calculation.
