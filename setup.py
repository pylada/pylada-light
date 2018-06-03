""" Pylada: A material science framework """

from os.path import dirname, join, abspath
from distutils.command.build import build as dBuild
from numpy.distutils.core import setup, Extension
from Cython.Build import cythonize


def description():
    """ Adds readme as description of package """
    source_dir = dirname(__file__)
    with open(join(source_dir, 'README.rst'), 'r') as readme:
        return readme.read()


def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    from tempfile import NamedTemporaryFile
    from distutils.errors import CompileError

    with NamedTemporaryFile('w', suffix='.cpp') as tempfile:
        tempfile.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([tempfile.name], extra_postargs=[flagname])
        except CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag.
    The c++14 is prefered over c++11 (when it is available).
    """
    if has_flag(compiler, '-std=c++11'):
        return '-std=c++11'
    else:
        raise RuntimeError('Unsupported compiler -- at least C++11 support '
                           'is needed!')


class Build(dBuild):
    """ Build that gets eigen """

    def run(self):
        """ Runs build after downloading eigen """
        from os import makedirs
        from os.path import exists
        from six.moves.urllib.request import urlretrieve
        import tarfile
        from shutil import move, rmtree
        makedirs(self.build_base, exist_ok=True)
        version = "3.3.4"
        url = "https://github.com/eigenteam/eigen-git-mirror/" \
            "archive/%s.tar.gz" % version
        finaldir = join(self.build_base, "eigen-include")
        if not exists(join(finaldir, "Eigen", "Core")):
            if exists(finaldir):
                rmtree(finaldir)
            tarpath = join(self.build_base, "eigen.tar.bz2")
            urlretrieve(url, tarpath)
            with tarfile.open(tarpath, "r:*") as tfile:
                tfile.extractall(self.build_base)
            move(
                join(self.build_base, "eigen-git-mirror-%s" % version),
                finaldir)

        dBuild.run(self)


setup(
    name="pylada",
    version="1.0",
    build_requires=["cython", "numpy"],
    install_requires=[
        'numpy', 'scipy', 'pytest', 'quantities', 'six', 'traitlets',
        'f90nml>=1.0', 'pytest-bdd'
    ],
    platforms=['GNU/Linux', 'Unix', 'Mac OS-X'],
    cmdclass={
        'build': Build,
    },
    author=["Peter Graf", "Mayeul d'Avezac"],
    author_email=["peter.graf@nrel.gov", "m.davezac@imperial.ac.uk"],
    description="Productivity environment for Density Functional Theory",
    license="GPL-2",
    url="https://github.com/pylada/pylada-light",
    ext_modules=cythonize([
        Extension(
            "pylada.ewald.utilities", [
                join("ewald", u) for u in
                ["cyewald.pyx"]
            ], include_dirs=[abspath(dirname(__file__))], language="c++"),
        Extension(
            "pylada.ewald.qq", [
                join("ewald", u) for u in
                ["ewald.h", "ewaldcc.cc", "ep_com.f90", "ewaldf.f90"]
            ], include_dirs=[abspath(dirname(__file__))], language="c++")
    ]),
    #  ext_modules=[
    #  Extension(u, []) for u in [
    #  'pylada.crystal.cutilities',
    #  'pylada.crystal.defects.cutilities',
    #  'pylada.decorations._cutilities',
    #  'pylada.ewald',
    #  ]
    #  ],
    package_dir={"pylada": dirname(__file__)},
    ext_package='pylada',
    packages=[
        'pylada', 'pylada.physics', 'pylada.jobfolder',
        'pylada.jobfolder.tests', 'pylada.crystal', 'pylada.crystal.tests',
        'pylada.crystal.defects', 'pylada.decorations',
        'pylada.decorations.tests', 'pylada.misc', 'pylada.config',
        'pylada.ipython', 'pylada.ipython.tests', 'pylada.ipython.launch',
        'pylada.ewald.tests', 'pylada.process', 'pylada.process.tests',
        'pylada.vasp', 'pylada.vasp.tests', 'pylada.vasp.extract',
        'pylada.vasp.extract.tests', 'pylada.vasp.nlep', 'pylada.vasp.incar',
        'pylada.vasp.incar.tests', 'pylada.tools', 'pylada.tools.tests',
        'pylada.tools.input', 'pylada.tools.input.tests', 'pylada.espresso',
        'pylada.espresso.tests'
    ],
    include_package_data=True,
    keywords="Physics",
    classifiers=[
        'Development Status :: 0 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries :: Application Frameworks',
    ],
    long_description=description())
