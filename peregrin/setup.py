# Always prefer setuptools over distutils
from setuptools import setup, find_packages

# This call to setup() does all the work
setup(
    name="peregrin",
    version="0.0.2",
    description="Demo",
    author="Branislav Modriansky",
    author_email="modrinaskybranislav@gmail.com",
    license="MIT",
    url="https://github.com/BranislavModriansky/Peregrin/tree/main",
    packages=find_packages(),
    include_package_data=True
)