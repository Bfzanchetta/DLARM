#!/bin/bash
#instalar java
rm /etc/apt/sources.list.d/webupd8team*
sudo add-apt-repository ppa:webupd8team/java
sudo add-apt-repository universe
sudo apt-get update
sudo apt-get install oracle-java8-installer -y
sudo apt-get install openjdk-8-jdk maven

#resto dos pacotes
sudo apt install -y liblapack3 libopenblas-base libopenblas-dev libatlas-dev libatlas-base-dev liblapack-dev
sudo apt install -y liblapacke-dev checkinstall # For OpenCV
sudo apt-get install autoconf automake libtool curl make cmake g++ unzip apt-utils git alien nano build-essential python-dev libopencv-dev graphviz python-pip
sudo apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-dev libhdf5-serial-dev protobuf-compiler libprotobuf-java
sudo apt-get install --no-install-recommends libboost-all-dev libboost-dev
sudo apt-get install -y libgflags-dev libgoogle-glog-dev liblmdb-dev
sudo pip install protobuf numpy scipy# ~20-40 min on HDD
sudo apt-get install -y libfreetype6-dev pkg-config libpng-dev libjpeg-dev zlib1g-dev
sudo apt-get install gfortran
sudo apt-get install -y python-matplotlib libcanberra-gtk-module
sudo pip install matplotlib==2.2.3
sudo pip install pyyaml scikit-build jupyter
sudo apt install -y libffi-dev
sudo pip install cffi pandas Cython scikit-image python-sklearn 
#se protobuf falhar, jogar pra ca

echo "export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:/usr/local/lib/python2.7/:$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc

sudo pip install gluoncv
