sudo apt-get install -y libsdl2-2.0-0 ffmpeg-doc ffmpeg libavdevice57 libsndfile1 libedit-dev
sudo apt-get install -y libgstreamer1.0-0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-doc gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio
sudo pip install Cython cffi PySoundFile
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-7.1.0/llvm-7.1.0.src.tar.xz
tar -xf llvm-7.1.0.src.tar.xz
cd llvm-7.1.0.src/
mkdir build
cd build/
cmake $LLVM_SRC_DIR -DCMAKE_BUILD_TYPE=Release \
                    -DLLVM_TARGETS_TO_BUILD="ARM;X86;AArch64"
