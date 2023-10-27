import os
import sys
from setuptools import setup, find_packages

PACKAGE_NAME = 'scapy'
MINIMUM_PYTHON_VERSION = 3, 8


def get_packages_required(include_version:bool = False) -> list:
    """Get all packages required by ScaPy.

    Args:
        include_version (bool, optional): Include versions. Defaults to False.

    Returns:
        list: List of packages required.
    """

    # Read all packages required by ScaPy
    with open('install_requires.txt', 'rb') as module:
        packages = module.read().decode('utf-16').split('\n')
        module.close

    packages_required = []   
    for item in packages:
        item = item.replace('\r', '')
        if item=='': continue
        if include_version:
            packages_required.append(item.strip())
        else:
            package, version = tuple(item.strip().split('=='))
            packages_required.append(package)

    return packages_required

def check_python_version():
    """Exit execution when the Python version is too low.
    """
    if sys.version_info < MINIMUM_PYTHON_VERSION:
        sys.exit("Python {}.{}+ is required.".format(*MINIMUM_PYTHON_VERSION))


def package_info(key:str) -> str:
    """Read the value of a variable from the package without importing.

    Args:
        key (str): Information to retrieve from the init file.

    Returns:
        str: Value of the information requested.
    """

    # Get the path of the module.
    module_path = os.path.join(PACKAGE_NAME, '__init__.py')

    # Read all lines in the module path.
    with open(module_path) as module:
        for line in module:
            parts = line.strip().split(' ')
            if parts and parts[0] == key:
                return parts[-1].strip("'")
    assert 0, "'{0}' not found in '{1}'".format(key, module_path)



check_python_version()

setup(
    name = 'scapy',
    version = package_info('__version__'),    
    description='Machine Learning for Spacecraft Conjunction Assessment optimisation.',
    # long_description='',
    url = 'https://github.com/JJavierRosales/scapy.git',
    author = 'Jose Javier Rosales Ruiz',
    author_email = 'javier.rosalesruiz@gmail.com',
    license = 'BSD',
    packages = find_packages(),
    install_requires = get_packages_required(False),
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],
)