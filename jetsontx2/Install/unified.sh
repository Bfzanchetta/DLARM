rm /etc/apt/sources.list.d/webupd8team*
sudo add-apt-repository ppa:webupd8team/java
sudo apt-get update
sudo apt-get install oracle-java8-installer -y
sudo apt-get install openjdk-8-jdk maven

apt-get install autoconf automake libtool curl make g++ unzip apt-utils
apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-dev libhdf5-serial-dev protobuf-compiler libprotobuf-java
apt-get install --no-install-recommends libboost-all-dev
apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev
apt-get install libatlas-base-dev libopenblas-dev
pip install protobuf
