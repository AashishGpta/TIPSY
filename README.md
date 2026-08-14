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
- check_env.py : Script to check that the installation and its dependencies work
- requirements.txt : Exact module versions used for the results in Gupta et al. (2024), for reproducing them

## Installation:
```
pip install git+https://github.com/AashishGpta/TIPSY.git
```
This installs TIPSY and all the modules it needs. `import tipsy` and `import trivia_my` then work from any directory.

If you do not want TIPSY installation to interfere with existing installations, I recommended to install TIPSY in a new Python environment:
```
conda create -n tipsyenv python=3.11
conda activate tipsyenv
pip install git+https://github.com/AashishGpta/TIPSY.git
```
TIPSY works with Python 3.9 and later. On Python 3.9, pip installs older versions of some modules, so Python 3.11 is recommended.

The `rebound_trajectory` function needs the [rebound](https://rebound.readthedocs.io) package, which is not installed by default because it is only an alternative to the Mendoza et al. (2009) trajectories. To include it:
```
pip install "tipsy[rebound] @ git+https://github.com/AashishGpta/TIPSY.git"
```

To modify the code, clone the repository and install it in editable mode, so that your edits take effect without reinstalling:
```
git clone https://github.com/AashishGpta/TIPSY.git
cd TIPSY
pip install -e .
```

## Checking the installation:

### Step 1: Run check_env.py
`check_env.py` runs the main TIPSY functions on a small synthetic streamer, which takes a few seconds and needs no observational data. It is a quick way to confirm that the installation works, and to check that a newer version of some module has not broken anything:
```
python check_env.py
```
It prints the version of every module used and a PASS/FAIL line per check. The script is in the repository, so run it from a clone (a pip install alone will not put it on your computer).

### Step 2: Run the tutorial notebooks
Once the checks pass, download the fits files for the S CrA and HL Tau tutorials from [here](https://virginia.box.com/s/4xwhmdh5gvsapybt5jurs5ey22i4kiuq).

Jupyter is not installed with TIPSY, you can install it within the same environment and start it from there:
```
pip install jupyterlab ipywidgets
jupyter lab
```
Jupyter can also be started from another environment, but then the TIPSY environment has to be registered as a kernel.

Now run `tipsy_scra_tutorial.ipynb` and `tipsy_hltau_tutorial.ipynb`. These notebooks show a complete fit on real observations and are the best starting point for using TIPSY on your own data.

## Legacy version:
The version of TIPSY before it became pip installable is kept on the `legacy` branch and the `legacy-v0` tag. It has to be set manually and needs Python 3.9:
```
git clone -b legacy https://github.com/AashishGpta/TIPSY.git
cd TIPSY
conda create -n tipsyenv_legacy python=3.9
conda activate tipsyenv_legacy
pip install -r requirements.txt
```
Note that `spectral-cube` is missing from `requirements.txt` in this version and has to be installed separately.

## Important things to keep in mind:
- TIPSY assumes that the protostar (center of gravity) is in the spatial center of the cube. This can be handled by centering the cube before loading it in TIPSY.
- Fitting results can be quite sensitive to the given systemic velocity of the source. Therefore, some care should be taken when estimating the systemic velocity (e.g., fitting gaussian to just the disk spectra from a non-absropbed tracer).

*If you encounter any other issues, or have suggestions for improvements, please open an issue here or send me a message at aashishgupta3008@gmail.com. I will try to address them in future updates.*
