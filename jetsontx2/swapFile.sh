fallocate -l 8G swapfile
ls -lh swapfile
chmod 600 swapfile
ls -lh swapfile
mkswap swapfile
sudo swapon swapfile
whereIsSwap=`swapon -s`
swap=`echo $whereIsSwap| cut -d " " -f 6`;
echo $swap"     none""     swap""     sw""     0""     0" >> /etc/fstab
    # swapfile_adress(including the filename)	none	swap   	sw    0    0
reboot
