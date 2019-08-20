sudo apt install -y libopenblas-dev liblapack-dev
sudo apt-get install -y libfreetype6-dev pkg-config
sudo apt-get install -y libjpeg-dev zlib1g-dev
sudo apt-get install -y autoconf automake libtool curl make g++ unzip cmake git alien nano python-dev
sudo apt install -y liblapacke-dev checkinstall # For OpenCV
sudo apt-get install python-pip
#pip install --upgrade pip==9.0.1
#sudo pip install numpy==1.15.1
sudo pip install numpy scipy # ~20-30 min
sudo apt-get install -y python-matplotlib
sudo pip install matplotlib==2.2.3
sudo pip install pyyaml
sudo pip install scikit-build
sudo apt-get -y install cmake
sudo apt install -y libffi-dev
sudo pip install cffi

sudo sh -c "echo export CUDNN_LIB_DIR=/usr/lib/aarch64-linux-gnu >> ~/.bashrc"
sudo sh -c "echo export CUDNN_INCLUDE_DIR=/usr/include >> ~/.bashrc"
source ~/.bashrc

wget https://rpmfind.net/linux/mageia/distrib/cauldron/aarch64/media/core/release/ninja-1.9.0-2.mga7.aarch64.rpm

sudo add-apt-repository universe
sudo apt-get update

sudo alien -g ninja-1.9.0-2.mga7.aarch64.rpm
cd ninja-1.9.0
sudo sed -i -- 's/aarch64/arm64/g' "debian/control"
# Previous line edits debian/control and replaces aarch64 to arm64 to allow compilation for platform
sudo debian/rules binary
cd ..
sudo dpkg -i ninja_1.9.0-3_arm64.deb
sudo rm -r ninja*

sudo apt install -y ninja-build
cd
git clone http://github.com/pytorch/pytorch
cd pytorch
sudo pip install -U setuptools
sudo pip install -r requirements.txt
git submodule update --init --recursive

sudo python setup.py build_deps

#If building pytorch 1.0.0 for TX2, then you may have to manually disable NCCL
#sudo gedit /pytorch/CMakeList.txt
#   > CmakeLists.txt : Change NCCL to 'Off' on line 97
#sudo gedit /pytorch/setup.py
#   > setup.py: Add USE_NCCL = False below line 198
#sudo gedit /pytorch/tools/setup_helpers/nccl.py
#   > nccl.py : Change USE_SYSTEM_NCCL to 'False' on line 8
#               Change NCCL to 'False' on line 78
#sudo gedit /pytorch/torch/csrc/cuda/nccl.h
#   > nccl.h : Comment self-include on line 8
#              Comment entire code from line 21 to 28
#sudo gedit torch/csrc/distributed/c10d/ddp.cpp
#   > ddp.cpp : Comment nccl.h include on line 6
#               Comment torch::cuda::nccl::reduce on line 148
   
sudo nvpmodel -m 0

export USE_NCCL=0
export USE_DISTRIBUTED=0
export TORCH_CUDA_ARCH_LIST="5.3;6.2"
export USE_OPENCV=ON
export USE_CUDNN=1
export USE_CUDA=1

sudo pip install scikit-build --user
sudo ldconfig

python setup.py bdist_wheel
sudo DEBUG=1 python setup.py build develop
sudo pip --no-cache-dir install torchvision

git clone https://github.com/python-pillow/Pillow.git
cd Pillow/
sudo python setup.py install
sudo pip install pandas Cython scikit-image python-sklearn  
sudo pip --no-cache-dir install torchvision
