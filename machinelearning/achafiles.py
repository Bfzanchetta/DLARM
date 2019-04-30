import glob, os

source_files = "C://Users//bfzan//Desktop//CNN//Dataset//Thumbnails//Process//tprocess6//Process6//"
destination_folder = "C://Users//bfzan//Desktop//CNN//Dataset//Thumbnails//Process//tprocess6//Process6//letter"

os.chdir(source_files)
for i in range (0, 20):
	aux = "slice_"+str(i)+"_*"
	aux1 = i+97
	aux2 = destination_folder+ str(chr(aux1))+ "//"
	for file in glob.glob(aux):
		os.rename(source_files+file,aux2+file)