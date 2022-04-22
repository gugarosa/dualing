from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="dualing",
    version="1.0.3",
    description="Dual-based Neural Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Gustavo Rosa",
    author_email="gustavo.rosa@unesp.br",
    url="https://github.com/gugarosa/dualing",
    license="Apache 2.0",
    install_requires=[
        "coverage>=5.5",
        "matplotlib>=3.3.4",
        "pre-commit>=2.17.0",
        "pylint>=2.7.2",
        "pytest>=6.2.2",
        "tensorflow<=2.6.2",
        "tensorflow-addons<=0.14.0",
    ],
    extras_require={
        "tests": [
            "coverage",
            "pytest",
            "pytest-pep8",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=find_packages(),
)
