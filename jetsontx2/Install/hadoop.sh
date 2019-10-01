#!/bin/bash
#Author: Breno Zanchetta
cd
mkdir BigData
echo "export BigData=~/BigData" >> ~/.bashrc
source ~/.bashrc

sudo apt-get update
sudo apt-get remove openjdk*

cd BigData
wget https://download.oracle.com/otn/java/jdk/8u112-b15/jdk-8u112-linux-arm64-vfp-hflt.tar.gz

tar -zxvf jdk-8u112-linux-arm64-vfp-hflt.tar.gz
sudo mkdir -p /opt/java
sudo mv jdk1.8.0_112 /opt/java
sudo update-alternatives --install "/usr/bin/java" "java" "/opt/java/jdk1.8.0_112/bin/java" 1
sudo update-alternatives --set java /opt/java/jdk1.8.0_112/bin/java

sudo apt install python python-dev rpm yum build-essential libfreetype6 libfreetype6-dev fontconfig fontconfig-config libfontconfig1-dev libssl-dev openssl findbugs -y
sudo apt-get install -y openssh* openssl* libssl* pkg-config* cmake* libsnappy-dev* bzip2* libbz2-dev* build-essential* autoconf* automake* libtool* zlib1g* libjansson* fuse*

cd BigData
wget https://github.com/google/protobuf/releases/download/v2.5.0/protobuf-2.5.0.tar.gz
tar xzvf protobuf-2.5.0.tar.gz
cd protobuf-2.5.0/
wget https://gist.github.com/BennettSmith/7111094/archive/40085b5022b5bc4d5656a9906aee30fa62414b06.zip
unzip 40085b5022b5bc4d5656a9906aee30fa62414b06.zip
cd 7111094-40085b5022b5bc4d5656a9906aee30fa62414b06
mv * ../
cd ..
git apply 0001-Add-generic-gcc-header-to-Makefile.am.patch
git apply 0001-Add-generic-GCC-support-for-atomic-operations.patch
./configure --prefix=/usr
make
make check
sudo make install

cd BigData
wget https://archive.apache.org/dist/maven/maven-3/3.0.5/binaries/apache-maven-3.0.5-bin.tar.gz
tar xvf apache-maven-3.0.5-bin.tar.gz
cd apache-maven-3.0.5/bin
echo "export PATH=$PWD:$PATH" >> ~/.bashrc
source ~/.bashrc

cd BigData
wget https://downloads.lightbend.com/scala/2.11.6/scala-2.11.6.tgz
tar -xf scala-2.11.6.tgz 

https://linuxconfig.org/how-to-install-hadoop-on-ubuntu-18-04-bionic-beaver-linux

#Install Deps for Ambari
sudo apt install python python-dev rpm yum build-essential libfreetype6 libfreetype6-dev fontconfig fontconfig-config libfontconfig1-dev libssl-dev openssl findbugs -y

