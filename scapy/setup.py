from setuptools import find_packages, setup

setup(
    name='irplib',
    packages=find_packages(include=['irplib']),
    version='0.1.0',
    description='Python library for the Individual Research Project',
    author='Jose Javier Rosales Ruiz',
    license='MIT',
    install_requires=['numpy', 'pandas', 'matplotlib', 'scipy', 'pytorch'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)