# Table of Package Versions for Building Frameworks #

| PACKAGE       | VERSION FOR MxNet |
| ------------- | ------------- |
| CUDA | 10.0.166-1 |
| cuDNN | 7.3.1.28-1+cuda10.0  |
| OpenCV | 3.3.1 |
| Protobuf | libprotoc 3.0.0 |

| PACKAGE       | VERSION FOR MxNet | VERSION FOR PyTorch |
| ------------- | ------------- | ------------- |
| CUDA |  |  |
| cuDNN |  |  |
| OpenCV |  |  |
| Num|  |  |
| Pip |  |  |
| numpy |  |  |
| scipy |  |  |
| matplotlib |  |  |
| scikit-image |  |  |
| python-sklearn |  |  |
| protobuf |  |  |
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
