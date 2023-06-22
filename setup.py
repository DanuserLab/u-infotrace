from setuptools import setup

import os
import re

with open("requirements.txt", "r") as fp:
    install_requires = list(fp.read().splitlines())

setup(name='InfoFlow',
	  version='0.1',
	  description='Calculate multiscale pixel spatiotemporal information flow for 2D videos',
	  author='Felix Y. Zhou',
	  packages=['InfoFlow'],
	  #package_dir={"": "InfoFlow"}, # directory containing all the packages (e.g.  src/mypkg, src/mypkg/subpkg1, ...)
	  include_package_data=True,
	  install_requires=install_requires,
)

