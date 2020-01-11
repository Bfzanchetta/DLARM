#!/bin/bash
# ########################################################################################################################## #
# Name: General Configuration Script                                                                                         #
# Description: This shell script intends to set a shape for package installation and configuration to avoid disorganization. #
# ########################################################################################################################## #
## to-do: split chunks of code into smaller scripts for further verification and invoke these algorithms here
# Routine to remove useless folders from root folder
cd && sudo rm -r Templates/ Public/ Videos/ Pictures/ Documents/ Music/ examples.desktop
## to-do: add swap script
sudo echo "sudo jetson_clocks" >> ~/.bashrc ; source ~/.bashrc
sudo dpkg --add-architecture arm64

# Routine to generate folder schematics
# Specifications: Generate folder trees and assign global variables as shortcuts to .bashrc
cd 
mkdir BLonDD ; cd BLonDD && echo "export A="`pwd` >> ~/.bashrc ; source ~/.bashrc;

cd $A ; mkdir DeepLearning && cd DeepLearning && echo "export B="`pwd` >> ~/.bashrc ; source ~/.bashrc;

cd $B ; mkdir Experiments && cd Experiments && echo "export B1="`pwd` >> ~/.bashrc ; source ~/.bashrc;
cd $B1 ; mkdir Datasets && cd Datasets && echo "export B1A="`pwd` >> ~/.bashrc ; source ~/.bashrc;
cd $B1 ; mkdir Logs && cd Logs && echo "export B1B="`pwd` >> ~/.bashrc ; source ~/.bashrc;
cd $B1B ; mkdir LogsApp && cd LogsApp && echo "export B1B1="`pwd` >> ~/.bashrc ; source ~/.bashrc;
cd $B1B ; mkdir LogsMonitor && cd LogsMonitor && echo "export B1B2="`pwd` >> ~/.bashrc ; source ~/.bashrc;
cd $B1 ; mkdir Settings && cd Settings && echo "export B1C="`pwd` >> ~/.bashrc ; source ~/.bashrc;
cd $B1 ; mkdir Models && cd Models && echo "export B1D="`pwd` >> ~/.bashrc ; source ~/.bashrc;
cd $B1 ; mkdir Checkpoints && cd Checkpoints && echo "export B1E="`pwd` >> ~/.bashrc ; source ~/.bashrc;
cd $B1 ; mkdir Utils && cd Utils && echo "export B1F="`pwd` >> ~/.bashrc ; source ~/.bashrc;

cd $B ; mkdir Install && cd Install && echo "export B2="`pwd` >> ~/.bashrc ; source ~/.bashrc;
cd $B2 ; mkdir Scripts && cd Scripts && echo "export B2A="`pwd` >> ~/.bashrc ; source ~/.bashrc;
cd $B2 ; mkdir Packages && cd Packages && echo "export B2B="`pwd` >> ~/.bashrc ; source ~/.bashrc;

cd $B ; mkdir Tools && cd Tools && echo "export B3="`pwd` >> ~/.bashrc ; source ~/.bashrc;

cd $A ; mkdir BigData && cd BigData && echo "export C="`pwd` >> ~/.bashrc ; source ~/.bashrc;

## to-do: contains problems on navigating through folders, must finish Big Data also.

# Routine to remove Python 2.7 and its dependencies and install Python 3.5 and Pip
sudo apt remove -y python3.6-minimal python2.7-minimal
sudo apt autoremove -y
sudo rm -r /usr/lib/python3/dist-packages
sudo rm -r /usr/lib/python2.7/dist-packages
sudo apt-get install -y build-essential checkinstall
sudo apt-get install -y libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev
cd /usr/src
sudo wget https://www.python.org/ftp/python/3.5.6/Python-3.5.6.tgz
sudo tar xzf Python-3.5.6.tgz
sudo rm Python-3.5.6.tgz
cd Python-3.5.6
sudo ./configure --enable-optimizations
sudo make altinstall
sudo make install
echo "alias python=python3.5" >> ~/.bashrc ; source ~/.bashrc
echo "alias python3=python3.5" >> ~/.bashrc ; source ~/.bashrc
echo "alias sudo='sudo '" >> ~/.bashrc ; source ~/.bashrc
sudo python3.5 /usr/local/lib/python3.5/site-packages/easy_install.py pip
sudo pip install --upgrade pip
source ~/.bashrc

# Routine to download and install packages at pre-determined folders
# MxNet from Source #
## to-do: add cmake installation and configuration script
## to-do: criar função que chama (sudo apt remove -y python3.6-minimal python2.7-minimal) entre todo par de comandos de instalação
sudo apt install -y liblapack3 libopenblas-base libopenblas-dev
sudo apt install -y liblapack3 libopenblas-base libopenblas-dev libatlas-base-dev liblapack-dev
sudo apt install -y liblapacke-dev checkinstall # For OpenCV
sudo apt-get install -y autoconf automake libtool curl make g++ unzip apt-utils git alien htop build-essential libopencv-dev graphviz
sudo apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-dev libhdf5-serial-dev protobuf-compiler libprotobuf-java
sudo pip3 install -U Cython
sudo pip3 install -U protobuf numpy
sudo apt-get install -y libfreetype6-dev pkg-config libpng-dev libjpeg-dev zlib1g-dev
sudo apt-get install -y gfortran
sudo apt-get install -y python3-matplotlib libcanberra-gtk-module
sudo pip3 install -U matplotlib==2.2.3
sudo apt install -y libffi-dev
##to-do: solve scipy installation error
cd $B2B;
wget https://github.com/scipy/scipy/releases/download/v1.3.0/scipy-1.3.0.tar.gz
sudo tar -xf scipy-1.3.0.tar.gz
sudo rm scipy-1.3.0.tar.gz
cd scipy-1.3.0/
sudo python3 setup.py install --user
sudo pip3 install -U cffi pandas scikit-image
sudo apt-get install -y python-sklearn
sudo apt install -y ninja-build
cd $B2B
#wget https://github.com/protocolbuffers/protobuf/releases/download/v2.6.1/protobuf-2.6.1.tar.gz
#tar -xf protobuf-2.6.1.tar.gz
#rm protobuf-2.6.1.tar.gz
#cd protobuf-2.6.1/
#mkdir -p pbc-aarch64
#cd pbc-aarch64
#proto links: https://github.com/protocolbuffers/protobuf/issues/3912
#proto links: https://criu.org/Build_protobuf
wget https://github.com/protocolbuffers/protobuf/releases/download/v3.11.2/protobuf-all-3.11.2.tar.gz
tar -xf protobuf-all-3.11.2.tar.gz
rm protobuf-all-3.11.2.tar.gz
cd protobuf-3.11.2/

mkdir -p pbc-aarch64
cd pbc-aarch64

# clone and install MxNet from source
cd $B2B
wget https://github.com/apache/incubator-mxnet/releases/download/1.4.1/apache-mxnet-src-1.4.1-incubating.tar.gz
tar -xf apache-mxnet-src-1.4.1-incubating.tar.gz
sudo rm apache-mxnet-src-1.4.1-incubating.tar.gz
cd apache-mxnet-src-1.4.1-incubating/

# Routine to download and install PyTorch, TensorFlow and Keras
apt-get install openjdk-8-jdk
## to-do: test the next two lines:
sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev
sudo pip3 install -U numpy grpcio absl-py py-cpuinfo psutil portpicker six mock requests gast h5py astor termcolor protobuf keras-applications keras-preprocessing wrapt google-pasta
#
cd $B2B;
wget https://github.com/lhelontra/tensorflow-on-arm/releases/download/v1.14.0/tensorflow-1.14.0-cp35-none-linux_aarch64.whl
sudo pip install -U tensorflow-1.14.0-cp35-none-linux_aarch64.whl
wget https://developer.download.nvidia.com/compute/redist/jp/v33/tensorflow-gpu/tensorflow_gpu-1.14.0+nv19.9-cp35-cp35m-linux_aarch64.whl
sudo pip install -U tensorflow_gpu-1.14.0+nv19.9-cp35-cp35m-linux_aarch64.whl
#tensorflow import error fix: https://github.com/tensorflow/tensorflow/issues/22342

#horovod example: https://github.com/horovod/horovod/blob/master/docs/mxnet.rst
