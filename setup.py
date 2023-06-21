from setuptools import setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="ml_project",
    packages=["src", "src.entities", "src.models", "src.features", "src.maps"],
    install_requires=required,
)
