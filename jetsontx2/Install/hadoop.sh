#!/bin/bash
#Author: Breno Zanchetta
#Based on Scripts of: Jorge Ximendes
#User needs to start the download of Java at Oracle, which requires Login, I suggest 8u221
#alternatively, can use this:
sudo apt-get update && apt-get remove openjdk*
wget https://download.oracle.com/otn/java/jdk/8u112-b15/jdk-8u112-linux-arm64-vfp-hflt.tar.gz

tar -zxvf jdk-8u112-linux-arm64-vfp-hflt.tar.gz
sudo mkdir -p /opt/java
sudo mv jdk1.8.0_112 /opt/java
sudo update-alternatives --install "/usr/bin/java" "java" "/opt/java/jdk1.8.0_112/bin/java" 1
sudo update-alternatives --set java /opt/java/jdk1.8.0_112/bin/java


wget https://archive.apache.org/dist/maven/maven-3/3.3.9/binaries/apache-maven-3.3.9-bin.tar.gz
tar -xf apache-maven-3.3.9-bin.tar.gz

sudo apt install python python-dev rpm yum build-essential libfreetype6 libfreetype6-dev fontconfig fontconfig-config libfontconfig1-dev libssl-dev openssl findbugs -y

sudo apt-get install scala

sudo apt-get install -y openssh* openssl* libssl* pkg-config* cmake* libsnappy-dev* bzip2* libbz2-dev* build-essential* autoconf* automake* libtool* zlib1g* libjansson* fuse*
wget https://github.com/google/protobuf/releases/download/v2.5.0/protobuf-2.5.0.tar.gz
tar xzvf protobuf-2.5.0.tar.gz
cd protobuf-2.5.0/
./configure --prefix=/usr
wget https://gist.github.com/BennettSmith/7111094/archive/40085b5022b5bc4d5656a9906aee30fa62414b06.zip
unzip 40085b5022b5bc4d5656a9906aee30fa62414b06.zip
cd 7111094-40085b5022b5bc4d5656a9906aee30fa62414b06
mv * ../
cd ..
git apply 0001-Add-generic-gcc-header-to-Makefile.am.patch
git apply 0001-Add-generic-GCC-support-for-atomic-operations.patch
make
make check
sudo make install
wget https://archive.apache.org/dist/hadoop/common/hadoop-2.7.2/hadoop-2.7.2-src.tar.gz
tar xzvf hadoop-2.7.2-src.tar.gz
sed -i "105i \ \ \ \ <additionalparam>-Xdoclint:none</additionalparam>" ./hadoop-2.7.2-src/pom.xml
cd hadoop-2.7.2-src/hadoop-common-project/hadoop-common/src
wget https://issues.apache.org/jira/secure/attachment/12570212/HADOOP-9320.patch
patch < HADOOP-9320.patch
cd ~/hadoop-2.7.2-src/

sudo mvn package -Pdist,native -DskipTests -Dtar
