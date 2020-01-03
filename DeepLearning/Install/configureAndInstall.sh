#!/bin/bash
# ########################################################################################################################## #
# Name: General Configuration Script                                                                                         #
# Description: This shell script intends to set a shape for package installation and configuration to avoid disorganization. #
# ########################################################################################################################## #

# Routine to remove useless folders from root folder
cd && sudo rm -r Templates/ Public/ Videos/ Pictures/ Documents/ Music/ examples.desktop
## to-do: add swap script
sudo echo "sudo jetson_clocks" >> ~/.bashrc ; source ~/.bashrc

# Routine to generate folder schematics
# Specifications: Generate folder trees and assign global variables as shortcuts to .bashrc
cd 
mkdir BLonDD ; cd BLonDD && echo "export A="`pwd` >> ~/.bashrc ; source ~/.bashrc;

cd $A ; mkdir DeepLearning && cd DeepLearning && echo "export B="`pwd` >> ~/.bashrc ; source ~/.bashrc;

cd $B ; mkdir Experiments && cd Experiments && echo "export B1="`pwd` >> ~/.bashrc ; source ~/.bashrc;
cd $B1 ; mkdir Datasets && cd Datasets && echo "export B1A="`pwd` >> ~/.bashrc ; source ~/.bashrc;
cd $B1 ; mkdir Logs && cd Logs && echo "export B1B="`pwd` >> ~/.bashrc ; source ~/.bashrc;
cd $B1B ; mkdir LogsApp && cd LogsApp && echo "export B1B1="`pwd` >> ~/.bashrc ; source ~/.bashrc;
cd $B1B ; mkdir LogsMonitor && cd LogsMonitor && echo "export B1B2="`pwd` >> ~/.bashrc ; source ~/.bashrc;
cd $B1 ; mkdir Settings && cd Settings && echo "export B1C="`pwd` >> ~/.bashrc ; source ~/.bashrc;
cd $B1 ; mkdir Models && cd Models && echo "export B1D="`pwd` >> ~/.bashrc ; source ~/.bashrc;
cd $B1 ; mkdir Checkpoints && cd Checkpoints && echo "export B1E="`pwd` >> ~/.bashrc ; source ~/.bashrc;
cd $B1 ; mkdir Utils && cd Utils && echo "export B1F="`pwd` >> ~/.bashrc ; source ~/.bashrc;

cd $B ; mkdir Install && cd Install && echo "export B2="`pwd` >> ~/.bashrc ; source ~/.bashrc;
cd $B2 ; mkdir Scripts && cd Scripts && echo "export B2A="`pwd` >> ~/.bashrc ; source ~/.bashrc;
cd $B2 ; mkdir Packages && cd Packages && echo "export B2B="`pwd` >> ~/.bashrc ; source ~/.bashrc;

cd $B ; mkdir Tools && cd Tools && echo "export B3="`pwd` >> ~/.bashrc ; source ~/.bashrc;

cd $A ; mkdir BigData && cd BigData && echo "export C="`pwd` >> ~/.bashrc ; source ~/.bashrc;

## to-do: contains problems on navigating through folders, must finish Big Data also.

# Routine to remove Python 2.7 and its dependencies and install Python 3.5 and Pip
sudo apt remove -y python3.6-minimal
sudo apt remove -y python2.7-minimal
sudo apt autoremove -y
sudo rm -r /usr/lib/python3/dist-packages
sudo rm -r /usr/lib/python2.7/dist-packages
sudo apt-get install -y build-essential checkinstall
sudo apt-get install -y libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev
cd /usr/src
sudo wget https://www.python.org/ftp/python/3.5.6/Python-3.5.6.tgz
sudo tar xzf Python-3.5.6.tgz
sudo rm Python-3.5.6.tgz
cd Python-3.5.6
sudo ./configure --enable-optimizations
sudo make altinstall
sudo make install
echo "alias python=python3.5" >> ~/.bashrc ; source ~/.bashrc
echo "alias python3=python3.5" >> ~/.bashrc ; source ~/.bashrc
sudo python3.5 /usr/local/lib/python3.5/site-packages/easy_install.py pip


# Routine to download and install packages at pre-determined folders
# MxNet from Source #
sudo apt install -y liblapack3 libopenblas-base libopenblas-dev
sudo apt-get install -y git build-essential libatlas-base-dev libopencv-dev graphviz








