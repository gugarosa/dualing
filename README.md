# Dualing: Dual-based Neural Learning

[![Latest release](https://img.shields.io/github/release/gugarosa/dualing.svg)](https://github.com/gugarosa/dualing/releases)
[![Build status](https://img.shields.io/travis/com/gugarosa/dualing/master.svg)](https://github.com/gugarosa/dualing/releases)
[![Open issues](https://img.shields.io/github/issues/gugarosa/dualing.svg)](https://github.com/gugarosa/dualing/issues)
[![License](https://img.shields.io/github/license/gugarosa/dualing.svg)](https://github.com/gugarosa/dualing/blob/master/LICENSE)

## Welcome to Dualing.

Have you ever wanted to find if there is any similarity between your data? If yes, Dualing is the right package! We implement state-of-the-art dual-based neural networks, such as Siamese Networks, to cope with learning similarity functions between sets of data. Such a strategy helps in providing clearer manifolds and better-embedded data for a wide range of applications.

Use Dualing if you need a library or wish to:

* Create similarity measures;
* Design or use pre-implement state-of-the-art Siamese Networks;
* Mix-and-match a new approach to solve your problem;
* Because it is fun to find resemblances;

Read the docs at [dualing.readthedocs.io](https://dualing.readthedocs.io).

Dualing is compatible with: **Python 3.6+**.

---

## Package guidelines

1. The very first information you need is in the very **next** section.
2. **Installing** is also easy if you wish to read the code and bump yourself into, follow along.
3. Note that there might be some **additional** steps in order to use our solutions.
4. If there is a problem, please do not **hesitate**. Call us.

---

## Getting started: 60 seconds with Dualing

First of all. We have examples. Yes, they are commented. Just browse to `examples/`, choose your subpackage, and follow the example. We have high-level examples for most of the tasks we could think.

Alternatively, if you wish to learn even more, please take a minute:

Dualing is based on the following structure, and you should pay attention to its tree:

```
- dualing
    - core
        - dataset
        - loss
        - model
    - datasets
        - batch
        - pair
    - models
        - base
            - cnn
            - mlp
        - contrastive
        - cross_entropy
        - triplet
    - utils
        - constants
        - exception
        - logging
        - projector
```

### Core

Core is the core. Essentially, it is the parent of everything. You should find parent classes defining the basis of our structure. They should provide variables and methods that will help to construct other modules.

### Datasets

Because we need data, right? Datasets are composed of classes and methods that allow preparing data for further application in dual-based learning.

### Models

This is the heart. All models are declared and implemented here. We will offer you the most fantastic implementation of everything we are working with. Please take a closer look at this package.

### Utils

This is a utility package. Common things shared across the application should be implemented here. It is better to implement once and use it as you wish than re-implementing the same thing repeatedly.

---

## Installation

We believe that everything has to be easy. Not tricky or daunting, Dualing will be the one-to-go package that you will need, from the very first installation to the daily-tasks implementing needs. If you may just run the following under your most preferred Python environment (raw, conda, virtualenv, whatever):

```Python
pip install dualing
```

Alternatively, if you prefer to install the bleeding-edge version, please clone this repository and use:

```Python
pip install .
```

---

## Environment configuration

Note that sometimes, there is a need for additional implementation. If needed, from here, you will be the one to know all of its details.

### Ubuntu

No specific additional commands needed.

### Windows

No specific additional commands needed.

### MacOS

No specific additional commands needed.

---

## Support

We know that we do our best, but it is inevitable to acknowledge that we make mistakes. If you ever need to report a bug, report a problem, talk to us, please do so! We will be available at our bests at this repository or gustavo.rosa@unesp.br.

---