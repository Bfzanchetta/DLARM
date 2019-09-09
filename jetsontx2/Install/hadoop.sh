#!/bin/bash
#Author: Breno Zanchetta
#Based on Scripts of: Jorge Ximendes
#User needs to start the download of Java at Oracle, which requires Login
#alternatively, can use this:
wget https://github.com/frekele/oracle-java/releases/download/8u212-b10/jdk-8u212-linux-arm64-vfp-hflt.tar.gz
tar -xf jdk-8u212-linux-arm64-vfp-hflt.tar.gz
cd jdk1.8.0_212/
cd bin/
echo "export JAVA_HOME="`pwd` >> ~/.bashrc
source ~/.bashrc
cd
sudo apt-get install -y openssh* openssl* libssl* pkg-config* cmake* libsnappy-dev* bzip2* libbz2-dev* build-essential* autoconf* automake* libtool* zlib1g* libjansson* fuse*
wget https://github.com/google/protobuf/releases/download/v2.5.0/protobuf-2.5.0.tar.gz
tar xzvf protobuf-2.5.0.tar.gz
cd protobuf-2.5.0/
./configure --prefix=/usr
