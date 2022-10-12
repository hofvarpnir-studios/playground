from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="playground",
    version="0.1",
    description="Example code for ML Training",
    author="Shauna Kravec, Nova DasSarma",
    author_email="shauna@hofvarpnir.ai, nova@hofvarpnir.ai",
    packages=["playground"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.8",
)
