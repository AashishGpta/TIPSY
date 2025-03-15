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
   
The fits files for S CrA and HL Tau tutorials can be found [here](https://virginia.box.com/s/4xwhmdh5gvsapybt5jurs5ey22i4kiuq).

## Recommended setup:
- Create a Python environment for TIPSY so that it does not interfere with existing installations: `conda create -n tipsyenv python=3.9`
- Activate the new environment: `conda activate tipsyenv`
- Install all the required modules: `pip install -r requirements.txt`

Then try to run the tutorial notebooks using Jupyter and have fun.  

## Some known issues:
- TIPSY assumes that the protostar (center of gravity) is in the spatial center of the cube. This can be handled by centering the cube before loading it in TIPSY. 
- Fitting results can be quite sensitive to the given systemic velocity of the source. Therefore, some care should be taken when estimating the systemic velocity (e.g., fitting gaussian to just the disk spectra from a non-absropbed tracer). 
  
_Please inform us if you find some other issues. We will try to address them in future updates._ 



