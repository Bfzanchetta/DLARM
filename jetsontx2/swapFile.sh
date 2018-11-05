fallocate -l 8G swapfile
ls -lh swapfile
chmod 600 swapfile
ls -lh swapfile
mkswap swapfile
sudo swapon swapfile
swapon -s
sudo gedit /etc/fstab
    > endereço_do_arquivo_swapfile(com o nome do arquivo)	none	swap	sw	0	0
reboot
