import os
import sys
import platform
import subprocess
import setuptools
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# Read long description from README
with open("../README.md", encoding="utf-8") as f:
    long_description = f.read()

# Version of the package
version = "0.1.0"

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the extension"
            )
            
        for ext in self.extensions:
            self.build_extension(ext)
            
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Make sure lib directory exists
        os.makedirs(extdir, exist_ok=True)
        
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE=Release",
            f"-DDEEPCFR_BUILD_TESTS=OFF",
            f"-DDEEPCFR_BUILD_BENCHMARKS=OFF",
            f"-DDEEPCFR_BUILD_PYTHON_BINDINGS=ON",
        ]
        
        build_args = [
            "--config", "Release",
            "--target", "deepcfr_cpp"
        ]
        
        if platform.system() == "Windows":
            cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            build_args += ["--", f"-j{os.cpu_count()}"]
            
        os.makedirs(self.build_temp, exist_ok=True)
        
        # CMake configure
        print("Configuring CMake project...")
        subprocess.check_call(
            ["cmake", os.path.join(ext.sourcedir, "..")] + cmake_args, 
            cwd=self.build_temp
        )
        
        # CMake build
        print("Building CMake project...")
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, 
            cwd=self.build_temp
        )
        
        print(f"Extension built successfully at {extdir}")

setup(
    name="deepcfr-cpp",
    version=version,
    author="DeepCFR Team",
    author_email="example@example.com",
    description="Deep CFR Poker AI C++ Implementation with Python Bindings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/deepcfr-cpp",
    packages=["deepcfr"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "torch>=2.0.0",
        "pokers-db>=2.2",
    ],
    ext_modules=[CMakeExtension("deepcfr")],
    cmdclass={
        "build_ext": CMakeBuild,
    }
)