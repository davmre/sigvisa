#!/bin/bash

source /home/sigvisa/.virtualenvs/sigvisa/bin/activate

export HOME=/home/sigvisa/

cd $HOME
pip install pyublas -d .
tar xvfz PyUblas-2013.1.tar.gz
cd PyUblas-2013.1/
echo "def use_setuptools(): pass" > distribute_setup.py
python setup.py install
cd $HOME
rm -r PyUblas*
