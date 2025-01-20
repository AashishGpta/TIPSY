# TIPSY
TIPSY stands for Trajectory of Infalling Particles in Streamers around Young stars.
It's a python code for fitting molecular-line observations of elongated structures, often called streamers, around young stars. The code fits such structures with theorectically expected trajectories of infalling gas, following the equations given in [Mendoza et al. (2019)](https://ui.adsabs.harvard.edu/abs/2009MNRAS.393..579M/abstract). For a complete description of the fitting methodology, refer to the [Gupta et al. (2024)](https://ui.adsabs.harvard.edu/abs/2024A%26A...683A.133G/abstract) paper.
The objective of TIPSY is to check:
1. If these structures are infalling streamers?
2. If yes, then how well can we characterise them?

## Desciption of main files:
- tipsy.py : Main file with codes for generating trajectories and fitting them   
- trivia_my.py : Code for visualising PPV diagrams, modified version of [TRIVIA](https://github.com/jaehanbae/trivia)   
- tipsy_scra_tutorial.ipynb : Notebook showing example of fitting streamer around SCrA using TIPSY 
- tipsy_hltau_tutorial.ipynb : Notebook showing example of fitting streamer around HL Tau using TIPSY 

## Some known issues:
- TIPSY assumes that the protostar (center of gravity) is in the spatial center of the cube.
- Distance units other than parsecs may not work properly.
- Fitting results can be quite sensitive to the given systemic velocity of the source.
_Please inform us if you find some other issues. We will try to address them in future updates._ 



