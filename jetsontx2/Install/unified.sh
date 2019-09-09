#!/bin/bash
#instalar java
rm /etc/apt/sources.list.d/webupd8team*
sudo add-apt-repository ppa:webupd8team/java
sudo add-apt-repository universe
sudo apt-get update
sudo apt-get install oracle-java8-installer -y
sudo apt-get install openjdk-8-jdk maven

#resto dos pacotes
sudo apt install -y liblapack3 libopenblas-base libopenblas-dev libatlas-dev libatlas-base-dev liblapack-dev
sudo apt install -y liblapacke-dev checkinstall # For OpenCV
sudo apt-get install -y autoconf automake libtool curl make cmake g++ unzip apt-utils git alien nano htop build-essential python-dev libopencv-dev graphviz python-pip
sudo apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-dev libhdf5-serial-dev protobuf-compiler libprotobuf-java
sudo apt-get install --no-install-recommends libboost-all-dev libboost-dev
sudo apt-get install -y libgflags-dev libgoogle-glog-dev liblmdb-dev
sudo pip install protobuf numpy scipy# ~20-40 min on HDD
sudo apt-get install -y libfreetype6-dev pkg-config libpng-dev libjpeg-dev zlib1g-dev
sudo apt-get install -y gfortran
sudo apt-get install -y python-matplotlib libcanberra-gtk-module
sudo pip install matplotlib==2.2.3
sudo pip install pyyaml scikit-build jupyter --user
sudo apt install -y libffi-dev
sudo pip install -y cffi pandas Cython scikit-image python-sklearn 
#se protobuf falhar, jogar pra ca

sudo alien -g ninja-1.9.0-2.mga7.aarch64.rpm
cd ninja-1.9.0
sudo sed -i -- 's/aarch64/arm64/g' "debian/control"
# Previous line edits debian/control and replaces aarch64 to arm64 to allow compilation for platform
sudo debian/rules binary
cd ..
sudo dpkg -i ninja_1.9.0-3_arm64.deb
sudo rm -r ninja*

sudo apt install -y ninja-build
sudo pip install -U setuptools
sudo nvpmodel -m 0
echo "export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:/usr/local/lib/python2.7/:$LD_LIBRARY_PATH" >> ~/.bashrc
sudo sh -c "echo export CUDNN_LIB_DIR=/usr/lib/aarch64-linux-gnu >> ~/.bashrc"
sudo sh -c "echo export CUDNN_INCLUDE_DIR=/usr/include >> ~/.bashrc"
source ~/.bashrc
sudo ldconfig

DIR="/sys/devices/system/cpu/cpu5/"
# If DIR exists, then the board is Jetson TX2 #
  echo "Installing MxNet for Jetson TX2."
  wget https://github.com/apache/incubator-mxnet/releases/download/1.5.1.rc0/apache-mxnet-src-1.5.1.rc0-incubating.tar.gz
  tar -xzvf apache-mxnet-src-1.5.1.rc0-incubating.tar.gz
  cd apache-mxnet-src-1.5.1.rc0-incubating
  sed -i 's/MSHADOW_CFLAGS += -DMSHADOW_USE_PASCAL=0/MSHADOW_CFLAGS += -DMSHADOW_USE_PASCAL=1/' ./3rdparty/mshadow/make/mshadow.mk
  sed -i 's/MSHADOW_LDFLAGS += -lblas/MSHADOW_LDFLAGS += -lblas \n        MSHADOW_CFLAGS += -DMSHADOW_USE_PASCAL=1/' ./3rdparty/mshadow/make/mshadow.mk
  cp make/config.mk .
  sed -i 's/USE_CUDA = 0/USE_CUDA = 1/' config.mk
  sed -i 's/USE_CUDA_PATH = NONE/USE_CUDA_PATH = \/usr\/local\/cuda/' config.mk
  sed -i 's/USE_CUDNN = 0/USE_CUDNN = 1/' config.mk
  sed -i '/USE_CUDNN/a CUDA_ARCH := -gencode arch=compute_53,code=sm_53 -gencode arch=compute_62,code=sm_62' config.mk
  
  sudo make -j 4
  #Install PyTorch
  export USE_NCCL=0
  export USE_DISTRIBUTED=1
  export TORCH_CUDA_ARCH_LIST="5.3;6.2"
  export USE_OPENCV=ON
  export USE_CUDNN=1
  export USE_CUDA=1
  
else
  # Else it is a TX1 or Nano
  echo "Installing MxNet for Jetson TX1/Nano."
  wget https://github.com/apache/incubator-mxnet/releases/download/1.5.1.rc0/apache-mxnet-src-1.5.1.rc0-incubating.tar.gz
  tar -xzvf apache-mxnet-src-1.5.1.rc0-incubating.tar.gz
  cd apache-mxnet-src-1.5.1.rc0-incubating
  sed -i 's/MSHADOW_CFLAGS += -DMSHADOW_USE_PASCAL=0/MSHADOW_CFLAGS += -DMSHADOW_USE_PASCAL=1/' ./3rdparty/mshadow/make/mshadow.mk
  sed -i 's/MSHADOW_LDFLAGS += -lblas/MSHADOW_LDFLAGS += -lblas \n        MSHADOW_CFLAGS += -DMSHADOW_USE_PASCAL=1/' ./3rdparty/mshadow/make/mshadow.mk
  cp make/config.mk .
  sed -i 's/USE_CUDA = 0/USE_CUDA = 1/' config.mk
  sed -i 's/USE_CUDA_PATH = NONE/USE_CUDA_PATH = \/usr\/local\/cuda/' config.mk
  sed -i 's/USE_CUDNN = 0/USE_CUDNN = 1/' config.mk
  sed -i '/USE_CUDNN/a CUDA_ARCH := -gencode arch=compute_53,code=sm_53' config.mk
  
  sudo make -j 4
  
  #Install PyTorch
  export USE_NCCL=0
  export USE_DISTRIBUTED=1
  export TORCH_CUDA_ARCH_LIST="5.3"
  export USE_OPENCV=ON
  export USE_CUDNN=1
  export USE_CUDA=1
fi

sudo pip install gluoncv
