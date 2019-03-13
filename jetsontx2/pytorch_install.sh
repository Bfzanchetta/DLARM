sudo apt install libopenblas-dev libatlas-dev liblapack-dev
sudo apt install liblapacke-dev checkinstall # For OpenCV
sudo apt-get install python-pip

pip install --upgrade pip==9.0.1
sudo apt-get install python-dev

sudo pip install numpy==1.15.1
sudo pip install scipy # ~20-30 min
sudo apt-get install python-matplotlib
sudo pip install matplotlib==2.2.3
sudo pip install pyyaml
sudo pip install scikit-build
sudo apt-get -y install cmake
sudo apt install libffi-dev
sudo pip install cffi

#add couple of lines to the end of  ~/.bashrc or ~/.profile
sudo gedit ~/.bashrc
export CUDNN_LIB_DIR=/usr/lib/aarch64-linux-gnu
export CUDNN_INCLUDE_DIR=/usr/include
source ~/.bashrc

wget https://rpmfind.net/linux/mageia/distrib/cauldron/aarch64/media/core/release/ninja-1.8.2-3.mga7.aarch64.rpm

sudo add-apt-repository universe
sudo apt-get update
sudo apt-get install alien
sudo apt-get install nano
sudo alien ninja-1.8.2-3.mga7.aarch64.rpm
   #If previous line fails, proceed to <$sudo dpkg -i ninja-1.8.2-3.mga7.aarch64.deb> 
   sudo alien -g ninja-1.8.2-3.mga7.aarch64.rpm
   cd ninja-1.8.2
   sudo nano debian/control
   #at architecture, add arm64 after aarch64 (aarch64, arm64)
   sudo debian/rules binary
cd ..
sudo dpkg -i ninja_1.8.2-4_arm64.deb
sudo apt install ninja-build
cd ..
git clone http://github.com/pytorch/pytorch
cd pytorch
sudo pip install -U setuptools
sudo pip install -r requirements.txt
git submodule update --init --recursive

python setup.py build_deps

sudo gedit /pytorch/CMakeList.txt
   > CmakeLists.txt : Change NCCL to 'Off' on line 97
sudo gedit /pytorch/setup.py
   > setup.py: Add USE_NCCL = False below line 198
sudo gedit /pytorch/tools/setup_helpers/nccl.py
   > nccl.py : Change USE_SYSTEM_NCCL to 'False' on line 8
               Change NCCL to 'False' on line 78
sudo gedit /pytorch/torch/csrc/cuda/nccl.h
   > nccl.h : Comment self-include on line 8
              Comment entire code from line 21 to 28
sudo gedit torch/csrc/distributed/c10d/ddp.cpp
   > ddp.cpp : Comment nccl.h include on line 6
               Comment torch::cuda::nccl::reduce on line 148
   
sudo nvpmodel -m 0
sudo DEBUG=1 python setup.py build develop

#If build fails before 100%
sudo python setup.py clean
sudo DEBUG=1 python setup.py build develop

#elif build fails after 100%
sudo python setup.py clean
sudo DEBUG=1 python setup.py develop

sudo apt clean
sudo apt-get install libjpeg-dev zlib1g-dev

git clone https://github.com/python-pillow/Pillow.git
cd Pillow/
sudo python setup.py install
sudo pip install pandas # ~20-30 min
sudo pip install Cython
sudo pip install scikit-image
sudo apt install python-sklearn  

sudo pip --no-cache-dir install torchvision

