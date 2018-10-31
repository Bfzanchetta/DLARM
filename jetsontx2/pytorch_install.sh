sudo apt install libopenblas-dev libatlas-dev liblapack-dev
sudo apt install liblapacke-dev checkinstall # For OpenCV
sudo apt-get install python-pip
sudo pip install numpy scipy # ~20-30 min
sudo pip install pyyaml
sudo pip install scikit-build
sudo apt-get -y install cmake
sudo apt install libffi-dev
sudo pip install cffi

#add couple of lines to the end of  ~/.bashrc or ~/.profile
sudo gedit ~/.bashrc
export CUDNN_LIB_DIR=/usr/lib/aarch64-linux-gnu
export CUDNN_INCLUDE_DIR=/usr/include

pip install --upgrade pip==9.0.1

download https://rpmfind.net/linux/mageia/distrib/cauldron/aarch64/media/core/release/ninja-1.8.2-3.mga7.aarch64.rpm

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

git clone http://github.com/pytorch/pytorch
cd pytorch
sudo pip install -U setuptools
sudo pip install -r requirements.txt
python setup.py build_deps
git submodule update --init --recursive
sudo nvpmodel -m 0
sudo NO_CUDA=1 DEBUG=1 python setup.py build develop


sudo apt clean
sudo apt-get install libjpeg-dev zlib1g-dev
git clone https://github.com/python-pillow/Pillow.git
cd Pillow/
sudo python setup.py install
sudo pip --no-cache-dir install torchvision

