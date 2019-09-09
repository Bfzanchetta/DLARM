#!/bin/bash
#Author: Breno Zanchetta
#Based on Scripts of: Jorge Ximendes
#User needs to start the download of Java at Oracle, which requires Login, I suggest 8u221
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
sed -i 's/<maven-bundle-plugin.version>2.5.0</maven-bundle-plugin.version>/<maven-bundle-plugin.version>2.5.0</maven-bundle-plugin.version> \n    <additionalparam>-Xdoclint:none</additionalparam>/' ./hadoop-2.7.2-src/pom.xml
<maven-bundle-plugin.version>2.5.0</maven-bundle-plugin.version>
