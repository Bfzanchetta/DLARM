#!/bin/bash
#Author: Kassiano J. Matteussi and Breno Zanchetta
#/home/nvidia/dataset/train is the folder where all train subfolders are
#labels1.txt is a file that contains all names of folders from imagenet_labels.txt
aux=`ls /home/nvidia/dataset/train`
for i in {1..500}; do
	forward=`cat /home/breno/Desktop/labels1.txt | cut -d " " -f $i| cut -d \" -f 2` 
	scp -o ConnectTimeout=100000 -r /home/nvidia/dataset/train/$forward sshuser@bigdlcluster-ssh.azurehdinsight.net:~/agoravai2
	done;
echo 'fim'
