from os.path import basename, dirname, join, abspath
from os import getcwd, chdir
from setuptools import setup, Extension
from distutils.command.build import build as dBuild
from setuptools.command.install import install as dInstall
from setuptools.command.build_ext import build_ext as dBuildExt
from setuptools.command.bdist_egg import bdist_egg as dBuildDistEgg
from setuptools.command.sdist import sdist as dSDist
from setuptools.command.egg_info import egg_info as dEggInfo
from setuptools.command.develop import develop as dDevelop
from distutils.dir_util import mkpath

source_dir = dirname(abspath(__file__))
package_dir = join(source_dir, 'pkg_install')
long_description = open(join(source_dir, 'README.rst'), 'r').read()
mkpath(package_dir)

def cmake_cache_line(variable, value, type='STRING'):
    return "set(%s \"%s\" CACHE %s \"\")\n" % (variable, value, type)

def cmake_executable():
    """ Path to cmake executable """
    from distutils.spawn import find_executable
    cmake = find_executable('cmake')
    if cmake is None:
        raise RuntimeError('Could not find cmake executable in path')
    return cmake

class Build(dBuild):
    """ Build that runs cmake. """

    def configure_cmdl(self, filename):
        """ Creates cmake command-line

            First puts variables into a cache file. This is safer that going through the
            command-line.
        """
        from sys import executable
        # other args
        other_args = [
            cmake_cache_line('nobins', 'TRUE', 'BOOL'),
            cmake_cache_line('PYTHON_EXECUTABLE', executable, 'PATH'),
            cmake_cache_line('PYTHON_BINARY_DIR', package_dir, 'PATH'),
            cmake_cache_line('CMAKE_BUILD_TYPE', 'Release', 'STRING'),
            '\n',
        ]

        with open(filename, 'w') as file: file.writelines(other_args)
        return ['-C%s' % filename]

    def _configure(self, build_dir):
        from distutils import log
        from distutils.spawn import spawn

        current_dir = getcwd()
        mkpath(build_dir)
        command_line = self.configure_cmdl(join(build_dir, 'Variables.cmake'))
        log.info(
                "CMake: configuring with variables in %s "
                % join(build_dir, 'Variables.cmake')
        )
        cmake = cmake_executable()

        try:
            chdir(build_dir)
            spawn([cmake] + command_line + [source_dir])
        finally: chdir(current_dir)

    def _build(self, build_dir):
        from distutils import log
        from distutils.spawn import spawn

        log.info("CMake: building in %s" % build_dir)
        current_dir = getcwd()
        cmake = cmake_executable()

        try:
            chdir(build_dir)
            spawn([cmake, '--build', '.'])
        finally: chdir(current_dir)

    def cmake_build(self):
        build_dir = join(dirname(abspath(__file__)), self.build_base)
        self._configure(build_dir)
        self._build(build_dir)

    def run(self):
        self.cmake_build()
        try:
            prior = getattr(self.distribution, 'running_binary', False)
            self.distribution.running_binary = True
            self.distribution.have_run['egg_info'] = 0
            dBuild.run(self)
        finally: self.distribution.running_binary = prior

class Install(dInstall):
    def run(self):
        from distutils import log
        self.distribution.run_command('build')
        current_cwd = getcwd()
        build_dir = join(dirname(abspath(__file__)), self.build_base)
        cmake = cmake_executable()
        pkg = abspath(self.install_lib)
        log.info("CMake: Installing package to %s" % pkg)
        try:
            chdir(build_dir)
            self.spawn([cmake,
                '-DPYTHON_PKG_DIR=\'%s\'' % pkg,
                '..'
            ])
            self.spawn([cmake, '--build', '.', '--target', 'install'])
        finally: chdir(current_cwd)

        try:
            prior = getattr(self.distribution, 'running_binary', False)
            self.distribution.running_binary = True
            self.distribution.have_run['egg_info'] = 0
            dInstall.run(self)
        finally: self.distribution.running_binary = prior

class BuildExt(dBuildExt):
    def __init__(self, *args, **kwargs):
        dBuildExt.__init__(self, *args, **kwargs)
    def run(self): pass

class BuildDistEgg(dBuildDistEgg):
    def __init__(self, *args, **kwargs):
        dBuildDistEgg.__init__(self, *args, **kwargs)
    def run(self):

        try:
            prior = getattr(self.distribution, 'running_binary', False)
            self.distribution.running_binary = True
            self.run_command('build')
            dBuildDistEgg.run(self)
        finally: self.distribution.running_binary = prior

class EggInfo(dEggInfo):
    def __init__(self, *args, **kwargs):
        dEggInfo.__init__(self, *args, **kwargs)
    def run(self):
        from setuptools.command.egg_info import manifest_maker
        from os import listdir
        which_template = 'MANIFEST.source.in'

        dist = self.distribution
        old_values = dist.ext_modules, dist.ext_package, \
            dist.packages, dist.package_dir
        if len(listdir(package_dir)) != 0  \
            and getattr(self.distribution, 'running_binary', True):
            which_template = 'MANIFEST.binary.in'
        else:
            dist.ext_modules, dist.ext_package = None, None
            dist.packages, dist.package_dir = None, None

        try:
            old_template = manifest_maker.template
            manifest_maker.template = which_template
            dEggInfo.run(self)
        finally:
            manifest_maker.template = old_template
            dist.ext_modules, dist.ext_package = old_values[:2]
            dist.packages, dist.package_dir = old_values[2:]

class Develop(dDevelop):
    def run(self):
        if not self.uninstall:
            build = self.distribution.get_command_obj("build")
            build.cmake_build()
        dDevelop.run(self)



class SDist(dSDist):
    def __init__(self, *args, **kwargs):
        dSDist.__init__(self, *args, **kwargs)
    def run(self):
        dist = self.distribution
        try:
            old_values = dist.ext_modules, dist.ext_package, \
                dist.packages, dist.package_dir
            dist.ext_modules, dist.ext_package = None, None
            dist.packages, dist.package_dir = None, None
            dSDist.run(self)
        finally:
            dist.ext_modules, dist.ext_package = old_values[:2]
            dist.packages, dist.package_dir = old_values[2:]
try:
    cwd = getcwd()
    chdir(source_dir)
    setup(
        name = "pylada",
        version = "1.0",

        install_requires = ['numpy', 'scipy', 'pytest', 'quantities', 'cython', 'mpi4py'],
        platforms = ['GNU/Linux','Unix','Mac OS-X'],

        zip_safe = False,
        cmdclass = {
            'build': Build, 'install': Install,
            'build_ext': BuildExt, 'bdist_egg': BuildDistEgg,
            'egg_info': EggInfo, 'develop': Develop
        },

        author = ["Peter Graf"],
        author_email = "peter.graf@nrel.gov",
        description = "Productivity environment for Density Functional Theory",
        license = "GPL-2",
        url = "https://github.com/pylada/pylada",
        ext_modules = [Extension(u, []) for u in [
            'pylada.crystal.cutilities',
            'pylada.crystal.defects.cutilities',
            'pylada.enum._cutilities',
            'pylada.ewald',
        ]],
        ext_package = 'pylada',
        packages = [
            'pylada',
            'pylada.physics',
            'pylada.jobfolder', 'pylada.jobfolder.tests',
            'pylada.crystal', 'pylada.crystal.tests',
            'pylada.crystal.defects',
            'pylada.enum', 'pylada.enum.tests',
            'pylada.misc',
            'pylada.config',
            'pylada.ipython', 'pylada.ipython.tests', 'pylada.ipython.launch',
            'pylada.ewald.tests',
            'pylada.process', 'pylada.process.tests',
            # 'pylada.vasp', 'pylada.vasp.tests', 'pylada.vasp.extract',
            # 'pylada.vasp.extract.tests', 'pylada.vasp.nlep', 'pylada.vasp.incar',
            # 'pylada.vasp.incar.tests',
            'pylada.tools', 'pylada.tools.tests', 'pylada.tools.input',
            'pylada.tools.input.tests'
        ],
        package_dir = {'': str(basename(package_dir))},
        include_package_data=True,

        keywords= "Physics",
        classifiers = [
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
        long_description = long_description
    )
finally: chdir(cwd)
