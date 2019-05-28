"""Setup PyLada."""
from os.path import dirname, join
from sys import platform

from setuptools import find_packages
from skbuild import setup

tests_require = ["pytest", "pytest-bdd"]
install_requires = [
    "numpy",
    "scipy",
    "quantities",
    "cython",
    "six",
    "traitlets",
    "f90nml>=1.0",
    "nbconvert",
    "nbformat",
    "ipykernel",
    "IPython",
]

cmake_args = []
if platform.lower() == "darwin":
    cmake_args.append("-DCMAKE_OSX_DEPLOYMENT_TARGET:STRING=10.9")

setup(
    name="Pylada",
    version="1.1.0",
    install_requires=install_requires,
    platforms=["GNU/Linux", "Unix", "Mac OS-X"],
    author=["Peter Graf", "Mayeul d'Avezac"],
    author_email=["peter.graf@nrel.gov", "mayeul.davezac@ic.ac.uk"],
    description="Productivity environment for Density Functional Theory",
    license="GPL-2",
    url="https://github.com/pylada/pylada",
    packages=find_packages(exclude="tests"),
    keywords="Physics",
    classifiers=[
        "Development Status :: 0 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Public License v2 (GPLv2)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 2.7",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
    ],
    long_description=open(join(dirname(__file__), "README.rst"), 'r').read(),
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    extras_require={"dev": tests_require},
    cmake_args=cmake_args,
    cmake_languages=("CXX", "Fortran"),
)
