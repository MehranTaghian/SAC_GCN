from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='torchgraphs',
    version='0.0.1',
    packages=find_packages(where='src'),
    package_dir={"": "src"},
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.md').read(),
    python_requires='>=3.7',
    install_requires=requirements
)
