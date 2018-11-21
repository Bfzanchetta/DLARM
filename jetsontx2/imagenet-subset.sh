#code by: dusty-nv
wget https://rawgit.com/dusty-nv/jetson-inference/master/tools/imagenet-subset.sh
chmod +x imagenet-subset.sh
mkdir 12_classes
./imagenet-subset.sh /opt/datasets/imagenet/ilsvrc12 12_classes
