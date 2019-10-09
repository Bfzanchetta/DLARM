#Code made by Dusty-nv
#https://github.com/dusty-nv/jetson-inference/blob/master/README.md
wget --no-check-certificate https://nvidia.box.com/shared/static/gzr5iewf5aouhc5exhp3higw6lzhcysj.gz -O ilsvrc12_urls.tar.gz
tar -xzvf ilsvrc12_urls.tar.gz
wget https://cdn.jsdelivr.net/gh/dusty-nv/jetson-inference@master/tools/imagenet-download.py
python imagenet-download.py ilsvrc12_urls.txt . --jobs 100 --retry 3 --sleep 0
