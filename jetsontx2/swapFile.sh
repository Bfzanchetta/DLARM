fallocate -l 8G swapfile
ls -lh swapfile
chmod 600 swapfile
ls -lh swapfile
mkswap swapfile
sudo swapon swapfile
swapon -s
sudo gedit /etc/fstab
    > swapfile_adress(including the filename)	none	swap   	sw    0    0
reboot
