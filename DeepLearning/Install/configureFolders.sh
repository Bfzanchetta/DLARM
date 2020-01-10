#!/bin/bash
# ########################################################################################################################## #
# Name: General Configuration Script                                                                                         #
# Description: This shell script intends to set a shape for package installation and configuration to avoid disorganization. #
# ########################################################################################################################## #
## to-do: split chunks of code into smaller scripts for further verification and invoke these algorithms here
# Routine to remove useless folders from root folder
cd && sudo rm -r Templates/ Public/ Videos/ Pictures/ Documents/ Music/ examples.desktop
## to-do: add swap script
sudo echo "sudo jetson_clocks" >> ~/.bashrc ; source ~/.bashrc
sudo dpkg --add-architecture arm64

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
