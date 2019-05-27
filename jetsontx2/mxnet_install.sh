wget https://github.com/apache/incubator-mxnet/releases/download/1.2.1/apache-mxnet-src-1.2.1-incubating.tar.gz
tar -xzvf apache-mxnet-src-1.2.1-incubating.tar.gz

sudo apt-get install -y git build-essential libatlas-base-dev libopencv-dev graphviz python-pip
sudo apt install libopenblas-dev libatlas-dev liblapack-dev
sudo apt install liblapacke-dev checkinstall # For OpenCV

sudo pip install numpy==1.16.2
pip install --upgrade pip==9.0.1
sudo pip install --upgrade setuptools
#good practice
sudo pip install protobuf
sudo apt-get install libboost-dev libboost-all-dev
#end parenthesis
sudo pip install numpy==1.15.2
sudo pip install graphviz jupyter
git clone https://github.com/apache/incubator-mxnet.git --branch v1.4.x --recursive
cd incubator-mxnet/
