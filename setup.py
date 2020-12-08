from setuptools import find_packages, setup

setup(
    name="lingcomp",
    version="0.1.0",
    author="Gabriele Sarti",
    author_email="gabriele.sarti996@gmail.com",
    description="Studying the many facets of linguistic complexity",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="NLP linguistic complexity psycholinguistics",
    license="MIT",
    url="https://github.com/gsarti/linguistic-complexity",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
        "dataclasses;python_version<'3.7'",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "seaborn",
        "h5py",
        "matplotlib",
        "syntaxgym;python_version>'3.6'",
        "torch==1.6.0",
        "farm==0.5.0",
        "xlrd",
        "prettytable",
        "tqdm",
        "transformers==3.3.1"
    ],
    python_requires=">=3.6.0"
)