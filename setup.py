from setuptools import setup, find_packages

setup(name="amer_dialect_id",
      version="0.1",
      packages=find_packages(where="src"),
      package_dir={"": "src"})
