import os
import sys
from setuptools import setup, find_packages
from Cython.Build import cythonize


if sys.version_info.major != 3:
    print("This Python is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))
    
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='mjmpc',
    version='1.0.0',
    packages=find_packages(),
    description='Model-Predictive Control (MPC) algorithms for gym environments in MuJoCo',
    long_description=read('README.md'),
    long_description_content_type="text/markdown",
    url='https://github.com/mohakbhardwaj/mjmpc.git',
    author='Mohak Bhardwaj',
    install_requires=[
        'click', 
        'gym>=0.13', 
        'mujoco_py>=2.0', 
        'tqdm', 
        'numpy',
        'colorlog',
        'tabulate',
        'pandas',
    ],
#    ext_modules = cythonize(["mjmpc/envs/gym_env_wrapper_cy.pyx"], annotate=True)
)
