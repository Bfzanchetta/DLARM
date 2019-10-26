#!/bin/bash
arquivo=file.txt
while IFS= read -r line
do
  train=echo "$line" | cut -d "," -f 2
  time=echo "$line" | cut -d "]" -f 2
done < "$arquivo"
