1) Apagar tudo pela aplicação Disks do Ubuntu.
2) Usar o comando $sudo fdisk -l  para descobrir a partição
3) $sudo mkfs.ext4 /dev/sdb1
4) Criar um diretório onde quer montar o disco: $mkdir tantofaz/
5) $chmod 757 tantofaz/
6) $sudo mount /dev/sdb1 tantofaz/
7) $sudo cp -ax <pasta de origem> <pasta de destino>
8) $sudo umount /dev/sdb1

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
