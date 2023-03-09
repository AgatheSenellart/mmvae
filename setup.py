from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="bivae",
    version="0.0.1",
    description="Multimodal variational Autoencoders in Python. Code for the ICML2023 submission.",
    
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
)
