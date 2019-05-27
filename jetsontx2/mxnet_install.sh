wget https://github.com/apache/incubator-mxnet/releases/download/1.2.1/apache-mxnet-src-1.2.1-incubating.tar.gz

sudo apt-get update
sudo apt-get install -y git build-essential libatlas-base-dev libopencv-dev graphviz python-pip
sudo pip install --upgrade pip
sudo pip install --upgrade setuptools
#good practice
sudo pip install protobuf
sudo apt-get install libboost-dev libboost-all-dev
#end parenthesis
sudo pip install numpy==1.15.2
sudo pip install graphviz jupyter
git clone https://github.com/apache/incubator-mxnet.git --branch v1.4.x --recursive
cd incubator-mxnet/
