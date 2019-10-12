#arm+ 2cpu+ 1 Denver
sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"
sudo sh -c "echo 1 > /sys/devices/system/cpu/cpu0/online"
sudo sh -c "echo 1 > /sys/devices/system/cpu/cpu1/online"
sudo sh -c "echo 0 > /sys/devices/system/cpu/cpu2/online"
sudo sh -c "echo 0 > /sys/devices/system/cpu/cpu3/online"
sudo sh -c "echo 1 > /sys/devices/system/cpu/cpu4/online"
sudo sh -c "echo 1 > /sys/devices/system/cpu/cpu5/online"
