"""Setup PyLada."""
from pathlib import Path
from sys import platform

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
    name="pylada",
    version="1.0",
    install_requires=install_requires,
    platforms=["GNU/Linux", "Unix", "Mac OS-X"],
    author=["Peter Graf"],
    author_email="peter.graf@nrel.gov",
    description="Productivity environment for Density Functional Theory",
    license="GPL-2",
    url="https://github.com/pylada/pylada",
    packages=[
        "pylada",
        "pylada.errors",
        # 'pylada.physicr', 'pylada.jobfolder',
        # 'pylada.jobfolder.tests', 'pylada.crystal', 'pylada.crystal.tests',
        # 'pylada.crystal.defects', 'pylada.decorations',
        # 'pylada.decorations.tests', 'pylada.misc', 'pylada.config',
        # 'pylada.ipython', 'pylada.ipython.tests', 'pylada.ipython.launch',
        # 'pylada.ewald.tests', 'pylada.process', 'pylada.process.tests',
        # 'pylada.vasp', 'pylada.vasp.tests', 'pylada.vasp.extract',
        # 'pylada.vasp.extract.tests', 'pylada.vasp.nlep',
        # 'pylada.vasp.incar', 'pylada.vasp.incar.tests', 'pylada.tools',
        # 'pylada.tools.tests', 'pylada.tools.input',
        # 'pylada.tools.input.tests', 'pylada.espresso',
        # 'pylada.espresso.tests'
    ],
    include_package_data=True,
    keywords="Physics",
    classifiers=[
        "Development Status :: 0 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 2.6",
        "Programming Language :: Python :: 2.7",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
    ],
    long_description=Path(__file__).parent.joinpath("README.rst").read_text(),
    setup_requires=["pytest-runner"],
    extras_require={"dev": tests_require},
    cmake_args=cmake_args,
    cmake_languages=("CXX", "Fortran"),
)
