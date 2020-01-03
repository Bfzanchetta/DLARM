https://github.com/lhelontra/tensorflow-on-arm/releases
https://github.com/lhelontra/tensorflow-on-arm/releases/tag/v2.0.0

wget http://developer.download.nvidia.com/compute/redist/jp33/tensorflow-gpu/tensorflow_gpu-1.9.0+nv18.8-cp27-cp27mu-linux_aarch64.whl
sudo add-apt-repository ppa:webupd8team/java
sudo apt-get update
sudo apt-get install oracle-java8-installer -y
sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev
pip install -U numpy grpcio absl-py py-cpuinfo psutil portpicker six mock requests gast h5py astor termcolor protobuf keras-applications keras-preprocessing wrapt google-pasta setuptools testresources
sudo pip install tensorflow_gpu-1.9.0+nv18.8-cp27-cp27mu-linux_aarch64.whl
sudo pip install tensorflow-datasets
