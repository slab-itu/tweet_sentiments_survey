import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="senti_altmetrics",
    version="0.0.1",
    author="Aneela Saleem",
    author_email="aneela.saleemh@itu.edu.pk",
    description="A sentiment analysis package for altmetrics dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/umeshdhakar/sample-python-package",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
) 
