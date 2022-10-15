import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    required = f.read().splitlines()

setuptools.setup(
    name="hpbandster-sklearn",
    version="2.0.2",
    author="Antoni Baum",
    author_email="antoni.baum@protonmail.com",
    description="A scikit-learn wrapper for HpBandSter hyper parameter search",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Yard1/hpbandster-sklearn",
    packages=setuptools.find_packages(include=['hpbandster_sklearn', 'hpbandster_sklearn.*']),
    install_requires=required,
    include_package_data=True,
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords=["distributed", "optimization", "multifidelity"],
)
