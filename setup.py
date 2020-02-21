from os import path

from setuptools import find_packages, setup

# META
##############################################################################
AUTHORS = "F. Dangel, F. Kunstner"
NAME = "backpack-for-pytorch"
PACKAGES = find_packages()

DESCRIPTION = r"""BACKpropagation PACKage"""
LONG_DESCR = """
    A backpack for PyTorch to compute quantities beyond the gradient.
    https://github.com/f-dangel/backpack
    """

VERSION = "1.1.0"
URL = "https://github.com/f-dangel/backpack"
LICENSE = "MIT"

# DEPENDENCIES
##############################################################################
REQUIREMENTS_FILE = "requirements.txt"
REQUIREMENTS_PATH = path.join(path.abspath(__file__), REQUIREMENTS_FILE)

with open(REQUIREMENTS_FILE) as f:
    requirements = f.read().splitlines()

setup(
    author=AUTHORS,
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCR,
    long_description_content_type="text/markdown",
    install_requires=requirements,
    url=URL,
    license=LICENSE,
    packages=PACKAGES,
    zip_safe=False,
    python_requires=">=3.5",
)
