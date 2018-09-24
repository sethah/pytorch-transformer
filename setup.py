import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytorch-transformer",
    version="0.0.1",
    author="Seth Hendrickson",
    description="Pytorch implementation of transformer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sethah/pytorch-transformer",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)