import os
import re
import sys
import sysconfig
import platform
import subprocess

from os.path import splitext
from glob import glob
from os.path import basename

from distutils.version import LooseVersion
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)',
                                         out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(
                cfg.upper(),
                extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]

            # CMake does not really respect CC / CXX environment variables. Anaconda on the other hand sets these variables
            # if a compiler is installed inside the enviroment. If they are set, then lets just use whatever is set in CC / CXX
            if "CC" in os.environ:
                cmake_args += ['-DCMAKE_C_COMPILER=' + os.environ["CC"]]
            
            if "CXX" in os.environ:
                cmake_args += ['-DCMAKE_CXX_COMPILER=' + os.environ["CXX"]]
            
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args,
                              cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args,
                              cwd=self.build_temp)
        print()  # Add an empty line for cleaner output

setup(
    name='ShrubEnsemble',
    version='0.1',
    url='https://github.com/sbuschjaeger/ShrubEnsembles',
    author='Sebastian Buschjäger',
    author_email='sebastian.buschjaeger@tu-dortmund.de',
    description='Ensemble Learning via (biased) proximal gradient descent implemented in Python and C++.',
    long_description='',
    # tell setuptools to look for any packages under 'src'
    packages=find_packages('src'),
    # tell setuptools that all packages will be under the 'src' directory
    # and nowhere else
    package_dir={'': 'src'},
    zip_safe=False,
    # add an extension module named 'python_cpp_example' to the package 
    # 'python_cpp_example'
    ext_modules=[CMakeExtension('se/CShrubEnsembleBindings')],
    # add custom build_ext command
    cmdclass=dict(build_ext=CMakeBuild),
    #include_package_data=True,
    #py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
)

# from setuptools import find_packages    
# from setuptools import setup

# import platform
# import os
# import multiprocessing
# import re
# import sys
# import subprocess
# from distutils.version import LooseVersion
# from setuptools import Extension
# from setuptools.command.build_ext import build_ext


# class CMakeExtension(Extension):
#     def __init__(self, name, sourcedir=''):
#         Extension.__init__(self, name, sources=[])
#         self.sourcedir = os.path.abspath(sourcedir)


# class CMakeBuild(build_ext):
#     def run(self):
#         try:
#             print("CALLING STUFF")
#             out = subprocess.check_output(['cmake', '--version'])
#         except OSError:
#             raise RuntimeError("CMake must be installed to build the following extensions: " +
#                                ", ".join(e.name for e in self.extensions))

#         if platform.system() == "Windows":
#             cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
#             if cmake_version < '3.1.0':
#                 raise RuntimeError("CMake >= 3.1.0 is required on Windows")

#         for ext in self.extensions:
#             self.build_extension(ext)

#     def build_extension(self, ext):
#         extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
#         cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
#                       '-DPYTHON_EXECUTABLE=' + sys.executable]

#         self.debug = False
#         cfg = 'Debug' if self.debug else 'Release'
#         build_args = ['--config', cfg]

#         if platform.system() == "Windows":
#             cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
#             if sys.maxsize > 2 ** 32:
#                 cmake_args += ['-A', 'x64']
#             build_args += ['--', '/m']
#         else:
#             cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
#             build_args += ['--', '-j', str(multiprocessing.cpu_count())]

#         env = os.environ.copy()
#         env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
#                                                               self.distribution.get_version())
#         if not os.path.exists(self.build_temp):
#             os.makedirs(self.build_temp)
#         subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
#         subprocess.check_call(['cmake', '--build', '.', '--target', 'Prime'] + build_args, cwd=self.build_temp)

