#This code enters the dataset folder and uncompresses all .tar training archives into
#individual folders with the same name as the tar file
#
#Author: Kassiano J. Matteussi
for i in `ls`; do echo "Extracting $i ..."; aux=`echo $i| cut -d "." -f1`;mkdir $aux; tar -xvf $i -C $aux; rm $i; done
