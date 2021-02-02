# Table of Package Versions for Building Frameworks #

| PACKAGE       | VERSION FOR MxNet (2019/20) |  VERSION for Build2021 |
| ------------- | ------------- | ------------- |
| CUDA | 10.0.166-1 | 10.2.89 |
| cuDNN | 7.3.1.28-1+cuda10.0 | 8.0.0 |
| OpenCV | 3.3.1 | 3.4.8 |
| Protobuf | libprotoc 3.0.0 | libprotoc 3.8.0 |
| NCCL | ----------- | 2.7.5-ga |
| oMPI | ----------- | 3.0.0 |

| PACKAGE       | VERSION FOR MxNet | VERSION FOR PyTorch |
| ------------- | ------------- | ------------- |
| Numa |  |  |
| Pip |  |  |
| numpy |  |  |
| scipy |  |  |
| matplotlib |  |  |
| scikit-image |  |  |
| python-sklearn |  |  |
| protobuf |  |  |
| *PACKAGES BELOW* | *ONLY USED* | *FOR AUDIO* |
| LLVM |  |  |
| llvmlite |  |  |

configureAndInstall.sh : Main script that explains detailed steps of configuring and installing MxNet, PyTorch and TensorFlow
configureFolders.sh : Shell script that build the folder structure for this project.
install_opencv.sh : OpenCV installation for Jetson TX1 and TX2.
mxnet_install.sh: Trustworthy script for MxNet installation on TX2 from source.
pytorch_install.sh: Trustworthy script for PyTorch installation on TX2 from source.
sound.sh: Guide to install Librosa and MxNet sound package.
tensorflow_install.sh: Old guide to install TensorFlow on TX2, needs review.
unified.sh: First attempt to unite mxnet_install.sh and pytorch_install.sh.


Horovod 0.16.2 Installation:
Requires: GCC version 5.X and above.

numpy 1.18.1

numpy<=1.15.2,>=1.8.2', 'requests>=2.20.0,<3', 'graphviz<0.9.0,>=0.8.1'

| PACKAGE       | tensorflow==1.14.0 | keras==2.2.4 | torch==1.1.0 | torchvision | pyspark | >=mxnet 1.4.1 |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| CUDA |  |  |  |  |  |  |
| cuDNN |  |  |  |  |  |  |
| OpenCV |  |  |  |  |  |  |
| numpy |  |  |  |  |  |  |
| scipy |  |  |  |  |  |  |
| matplotlib |  |  |  |  |  |  |
| scikit-image |  |  |  |  |  |  |
| python-sklearn |  |  |  |  |  |  |
| protobuf |  |  |  |  |  |  |

https://github.com/AastaNV/JEP/tree/master/script/TensorFlow_1.6
http://comp.photo777.org/
https://gist.github.com/dusty-nv/ef2b372301c00c0a9d3203e42fd83426
https://horovod.readthedocs.io/en/latest/contributors_include.html
