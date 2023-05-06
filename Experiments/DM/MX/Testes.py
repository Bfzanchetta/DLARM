#!/bin/bash
for i in `ls *.py`; do
        echo 3 > /proc/sys/vm/drop_caches;
        python $i > $i.txt &
        sleep 30s;
        pkill python;
        done;
