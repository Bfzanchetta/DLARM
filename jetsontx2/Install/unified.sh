#!/bin/bash
#instalar java
rm /etc/apt/sources.list.d/webupd8team*
sudo add-apt-repository ppa:webupd8team/java
sudo apt-get update
sudo apt-get install oracle-java8-installer -y
sudo apt-get install openjdk-8-jdk maven

#resto dos pacotes
sudo apt install -y liblapack3 libopenblas-base libopenblas-dev libatlas-dev libatlas-base-dev 
sudo apt install -y liblapacke-dev checkinstall # For OpenCV
sudo apt-get install autoconf automake libtool curl make g++ unzip apt-utils git build-essential libopencv-dev graphviz python-pip
sudo apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-dev libhdf5-serial-dev protobuf-compiler libprotobuf-java
sudo apt-get install --no-install-recommends libboost-all-dev
sudo apt-get install -y libgflags-dev libgoogle-glog-dev liblmdb-dev
sudo pip install protobuf numpy# ~10-20 min on HDD
sudo apt-get install libfreetype6-dev pkg-config libpng-dev
sudo apt-get install gfortran
sudo pip install scipy# ~20-30 min on HDD
sudo apt-get install -y python-matplotlib libcanberra-gtk-module


echo "export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:/usr/local/lib/python2.7/:$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc
