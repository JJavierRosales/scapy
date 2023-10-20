from setuptools import find_packages, setup

setup(
    name='scapy',
    packages=find_packages(include=['scapy']),
    version='0.1.0',
    description='Machine Learning library for Spacecraft Conjunction ' + \
                'Assessment optimisation',
    author='Jose Javier Rosales Ruiz',
    license='GNU',
    install_requires=['os', 'numpy', 'pandas', 'matplotlib', 
                      'scipy', 'pytorch', 'sklearn'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)