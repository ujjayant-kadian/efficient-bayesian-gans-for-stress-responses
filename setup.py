from setuptools import setup, find_packages

setup(
    name="bayesian-gan-stress",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "matplotlib",
        "tqdm",
    ],
    python_requires=">=3.6",
) 