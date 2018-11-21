#!/bin/bash
#original code by: dusty-nv
#new code by: Breno Zanchetta (bfzanchetta)
#thanks to Kassiano Matteussi

#for .tar imagenet:
#for i in `ls`; do aux=`echo $i| cut -d "." -f1`; echo -n "\"$aux\" "; done >> output.txt
#
#for folders:
for i in `ls`; do echo -n "\"$aux\" "; done >> output.txt

#array=("A" "B" "ElementC" "ElementE")
array=(`imagenet_labels.txt`)
echo array

#for element in "${array[@]}"


#wget https://rawgit.com/dusty-nv/jetson-inference/master/tools/imagenet-subset.sh
#chmod +x imagenet-subset.sh
#mkdir 12_classes
#./imagenet-subset.sh /opt/datasets/imagenet/ilsvrc12 12_classes
#mkdir subset_training_folder
#mv imagenet-subset.sh


#for i in `ls`; do echo "Extracting $i ..."; aux=`echo $i| cut -d "." -f1`;mkdir $aux; tar -xvf $i -C $aux; done
