#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages


setup(name='fuzzy_clust_algos',
      version="1.0",
      description='Fuzzy Clustering Algorithms.',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'scipy',
      ],
      )
