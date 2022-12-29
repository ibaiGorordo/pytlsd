# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
from pathlib import Path

__version__ = "0.0.2"

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

ext_modules = [
    Pybind11Extension("pytlsd",
                      ["src/lsd.cpp",
                       "src/PYAPI.cpp"],
                      # Example: passing in the version to the compiled code
                      define_macros=[('VERSION_INFO', __version__)],
                      ),
]

# The information here can also be placed in setup.cfg - better separation of
# logic and declaration, and simpler if you include description/version in a file.
setup(
    name="pytlsd",
    version=__version__,
    author="Iago Suarez",
    author_email="iagoh92@gmail.com",
    description="Transparent bindings of LSD (Line Segment Detector)",
    long_description=long_description,
    long_description_content_type='text/markdown',
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    # Currently, build_ext only provides an optional "highest supported C++
    # level" feature, but in the future it may provide more features.
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
)
