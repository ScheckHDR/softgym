from setuptools import setup, find_packages

packages = find_packages()
print(packages)
setup(
    name="softgym",
    description="Softgym simulation environment.",
    packages=packages
)