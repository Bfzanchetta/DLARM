#!/bin/bash
sudo apt-get update
sudo apt-get install build-essential

wget https://cmake.org/files/v3.12/cmake-3.12.2.tar.gz
tar xf cmake-3.12.2.tar.gz
rm cmake-3.12.2.tar.gz
cd cmake-3.12.2/
./configure
sudo make -j 4
sudo make install

sudo apt-get install -y g++ freeglut3-dev build-essential libx11-dev \
    libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev

wget https://archive.apache.org/dist/maven/maven-3/3.6.0/binaries/apache-maven-3.6.0-bin.tar.gz -P /tmp
sudo tar xf /tmp/apache-maven-*.tar.gz -C /opt
sudo ln -s /opt/apache-maven-3.6.0 /opt/maven
sudo echo "" > /etc/profile.d/maven.sh
sudo echo "export JAVA_HOME=/opt/java/jdk1.8.0_221" >> /etc/profile.d/maven.sh
sudo echo "export M3_HOME=/opt/maven" >> /etc/profile.d/maven.sh
sudo echo "export MAVEN_HOME=/opt/maven" >> /etc/profile.d/maven.sh
sudo echo "export PATH=${M3_HOME}/bin:${PATH}" >> /etc/profile.d/maven.sh
sudo chmod +x /etc/profile.d/maven.sh
source /etc/profile.d/maven.sh

wget https://downloads.lightbend.com/scala/2.11.6/scala-2.11.6.tgz -P /tmp
sudo tar xvf /tmp/scala-2.11.6.tgz -C /opt
sudo ln -s /opt/scala-2.11.6/ /opt/scala
sudo echo "" > /etc/profile.d/scala.sh
sudo echo "export JAVA_HOME=/opt/java/jdk1.8.0_221" >> /etc/profile.d/scala.sh
sudo echo "export SCALA_HOME=/opt/scala" >> /etc/profile.d/scala.sh
sudo echo "export PATH=${SCALA_HOME}/bin:${PATH}" >> /etc/profile.d/scala.sh
sudo chmod +x /etc/profile.d/scala.sh
source /etc/profile.d/scala.sh

wget http://developer.download.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81.00_linux.run
sudo chmod +x cuda_9.0.176_384.81.00_linux.run
sudo sh cuda_9.0.176_384.81.00_linux.run 

#http://www.askaswiss.com/2019/01/how-to-install-cuda-9-cudnn-7-ubuntu-18-04.html

sudo apt-get install -y libgomp1
sudo apt-get install -y libopenblas-dev
echo "export PATH=$PATH:/usr/lib/x86_64-linux-gnu/openblas" >> ~/.bashrc
source ~/.bashrc

sudo add-apt-repository universe
sudo add-apt-repository main
sudo apt-get update
sudo apt-get install -y libatlas-base-dev liblapack-dev libblas-dev
sudo apt-get install -y libomp-dev

git clone --depth 1 --branch master https://github.com/deeplearning4j/dl4j-test-resources
cd dl4j-test-resources
mvn install
mvn clean test -P  testresources,test-nd4j-native

echo "export LIBND4J_HOME=/home/bigdldl4j/Pacotes/libnd4j" >> ~/.bashrc
source ~/.bashrc
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/mkl/lib/ia32:/opt/intel/lib/ia32"
source ~/.bashrc
echo "export MKL_THREADING_LAYER=GNU" >> ~/.bashrc
echo "export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1:/lib64/ld-linux-x86-64.so.2" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/libopenblas.so.0:/usr/lib/x86_64-linux-gnu/:/usr/lib/x86_64-linux-gnu/openblas/" >> ~/.bashrc
source ~/.bashrc


rm -rf libnd4j
rm -rf nd4j
rm -rf datavec
rm -rf deeplearning4j
git clone https://github.com/deeplearning4j/libnd4j.git
cd libnd4j
./buildnativeoperations.sh -c cuda -cc 37
echo "export LIBND4J_HOME=`pwd`" >> ~/.bashrc
source ~/.bashrc

git clone https://github.com/deeplearning4j/nd4j.git
cd nd4j
mvn clean install -DskipTests -Dmaven.javadoc.skip=true -pl '!:nd4j-tests'
#git clone https://github.com/eclipse/deeplearning4j.git

