#!/bin/bash

cd src/problems/tsp/cython_merge
python setup.py build_ext --inplace
cd ../../../..