<div align="center">
  <a> <img height="200px" src="docs/scapy-logo-named-cropped.png"></a>
</div>


# ***Spacecraft Conjunction Assessment optimisation library for Python*** 
 
A Machine learning library developed in Python library for ***Spacecraft Conjunction Assessment optimisation using Artificial Neural Networks on Conjunction Data Messages***.



## Authors

* José Javier Rosales Ruiz [[j.rosales-ruiz@cranfield.ac.uk](mailto:j.rosales-ruiz@cranfield.ac.uk)] <sup>a</sup>
* Nicola Garzaniti [[nicola.garzaniti@cranfield.ac.uk](mailto:nicola.garzaniti@cranfield.ac.uk)] <sup>b</sup>

<sup>a</sup> *Corresponding author*\
<sup>b</sup> *Supervisor*

## License

ScaPy is distributed under the [GNU General Public License version 3](LICENSE), whose terms are included in this repository.


## Installation



### Prerequisites

Since ScaPy uses [PyTorch](https://pytorch.org/get-started/locally/) for the deep learning models, the requirements differ depending on the platform as follows:

| Platform  | Requisites  | 
|:---:      |:---|
| Windows   | Python 3.8-3.11 (limited by PyTorch); Python 2.x is not supported. | 
| MacOS     | Python 3.8 or greater. |
| Linux     | Python 3.8 or greater. |

### How to install the library

#### Download repository
To download ScaPy, open the terminal (PowerShell in Windows) and run the following commands to clone repository and change working directory to the root of the project directory:
 ```
git clone https://github.com/JJavierRosales/scapy.git
cd scapy
```

#### Installing ScaPy on a Conda environment (OPTIONAL)

To create a Conda environment with a specific Python version, open your terminal and run the following command:
```
conda activate
conda create -n scapy-env python=3.11
conda activate scapy-env
```

<!-- **Note**: This repository also includes a `.yml` file containing all the conda environment settings required to run ScaPy. You can import the environment provided running the following command on your terminal (make sure the working directory where you run the command is the root directory of the project):
```
conda env create -n scapy-env --file conda-environment.yml
```
To export your conda environment run the following command:
```
conda activate scapy-env
conda env export --from-history > conda-environment.yml
``` -->
#### Installing ScaPy using Package Installer for Python (PIP)
PIP offers two modes to install Python projects: an ***editable***/***development*** mode -recommended for development- and a ***user*** mode. When installed as editable, the project can be edited in-place without reinstallation: changes to Python source files will be reflected the next time the interpreter process is started. To install ScaPy in *editable*/*development* mode, you can add  as follows:
```
python -m pip install -e .
```
where the PIP command-line flag `-e` stands for editable mode (short for `--editable`) and `.` indicate working directory.

Alternatively, if no modification is planned to be made after installation, the library can be installed using PIP *user* mode as follows:
```
python -m pip install .
```

To uninstall the library run the following command:
```
python -m pip uninstall scapy
```

## Get started with ScaPy

This repository includes the following Jupyter notebooks to support users to use the library:

 - [Using ScaPy without installation](notebooks/users/01_get_started_cef.ipynb)
 - [Forecast conjunctions evolutions](notebooks/users/01_get_started_cef.ipynb)
 - [Evaluate collision risk per conjunction](notebooks/users/02_get_started_cre.ipynb)
 - [Generate synthetic CDM data](notebooks/users/03_get_started_sdg.ipynb)

## To-Do
 - Method `evaluate()`  for collision risk evaluation. 
 - Section on how to use CRE to evaluate conjunctions in users notebook.
 - Learning rate scheduler for models training.
 - Docker container.
 - ...

