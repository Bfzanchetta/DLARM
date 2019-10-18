
#!/bin/bash
sudo apt-get update
echo -e "blacklist nouveau\nblacklist lbm-nouveau\noptions nouveau modeset=0\nalias nouveau off\nalias lbm-nouveau off\n" | sudo te$
echo options nouveau modeset=0 | sudo tee -a /etc/modprobe.d/nouveau-kms.conf
sudo update-initramfs -u
sudo apt-get install build-essential
sudo apt-get update
sudo apt purge gcc-6
sudo apt purge g++-6
sudo apt-get install gcc-4.8 g++-4.8 -y && sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 60 --slave /usr/bin$
sudo update-alternatives --config gcc

wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run
sudo sh cuda_8.0.61_375.26_linux-run --override --no-opengl-lib
./cuda*-run --tar mxvf
sudo cp InstallUtils.pm /usr/lib/x86_64-linux-gnu/perl-base
sudo sh cuda_8.0.61_375.26_linux-run --override --no-opengl-lib
sudo ldconfig

#wget https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v6/prod/8.0_20170427/Ubuntu16_04_x64/libcudnn6-dev_6.0.21-$
#sudo dpkg -i libcudnn6_6.0.21-1+cuda8.0_amd64.deb
#sudo dpkg -i libcudnn6-dev_6.0.21-1+cuda8.0_amd64.deb

wget https://github.com/eclipse/deeplearning4j/archive/_old/deeplearning4j-0.9.1.tar.gz
tar -xf deeplearning4j-0.9.1.tar.gz
rm deeplearning4j-0.9.1.tar.gz
cd deeplearning4j-_old-deeplearning4j-0.9.1/
