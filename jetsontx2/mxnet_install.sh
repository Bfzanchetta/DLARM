wget https://github.com/apache/incubator-mxnet/releases/download/1.2.1/apache-mxnet-src-1.2.1-incubating.tar.gz
tar -xzvf apache-mxnet-src-1.2.1-incubating.tar.gz

sudo apt-get install -y git build-essential libatlas-base-dev libopencv-dev graphviz python-pip
sudo apt install libopenblas-dev libatlas-dev liblapack-dev
sudo apt install liblapacke-dev checkinstall # For OpenCV
sudo pip install --upgrade pip==9.0.1
sudo pip install numpy==1.16.2

sudo pip install scipy # ~20-30 min
sudo apt-get install python-matplotlib
sudo pip install matplotlib==2.2.3
sudo pip install pyyaml
sudo pip install scikit-build
sudo apt-get -y install cmake
sudo apt install libffi-dev
sudo pip install cffi
sudo pip install pandas # ~20-30 min
sudo pip install Cython
sudo pip install scikit-image
sudo apt install python-sklearn 

#good practice
sudo pip install protobuf
sudo apt-get install libboost-dev libboost-all-dev
#end parenthesis
sudo pip install graphviz jupyter
#git clone https://github.com/apache/incubator-mxnet.git --branch v1.4.x --recursive
cd incubator-mxnet/

sudo gedit 3rdparty/mshadow/make
sed -i 's/MSHADOW_LDFLAGS += -lblas/MSHADOW_LDFLAGS += -lblas \n        MSHADOW_CFLAGS += -DMSHADOW_USE_PASCAL=1/' /home/dlarm2/apache-mxnet-src-1.2.1-incubating/3rdparty/mshadow/make/mshadow.mk

cp make/config.mk .
sed -i 's/USE_CUDA = 0/USE_CUDA = 1/' config.mk
sed -i 's/USE_CUDA_PATH = NONE/USE_CUDA_PATH = \/usr\/local\/cuda/' config.mk
sed -i 's/USE_CUDNN = 0/USE_CUDNN = 1/' config.mk
sed -i '/USE_CUDNN/a CUDA_ARCH := -gencode arch=compute_53,code=sm_53 -gencode arch=compute_62,code=sm_62' config.mk

pip install -U numpy 

sudo apt-get install gcc-6 g++-6
sed -i 's/export CC = gcc/export CC = gcc-6/' config.mk
sed -i 's/export CXX = g++/export CXX = g++-6/' config.mk
