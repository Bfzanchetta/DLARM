#!/bin/bash

var1=`grep -n "#from gluoncv.data import ImageNet" MX-GPU-L-INC-mIMGN10.py | cut -d ':' -f 1`;
#from multiprocessing import cpu_count
#https://mxnet.incubator.apache.org/versions/master/tutorials/gluon/gluon_from_experiment_to_deployment.html
sys.path.append('..')
import utils


echo $var1
sed $var1'd' 'MX-GPU-L-INC-mIMGN10.py'
