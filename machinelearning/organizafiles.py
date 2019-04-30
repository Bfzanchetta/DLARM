import glob, os

source_files = ""
destination_folder = "C://Users//bfzan//Desktop//CNN1//Dataset//Thumbnails//Final//letter"

ocorrencias = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#modelo     = [a,b,c,d,e,.........................,s,t]

for k in range(2, 7):
	print("Debugger")
	for j in range(0,20):
		source_files = "C://Users//bfzan//Desktop//CNN1//Dataset//Thumbnails//Process//tprocess"+str(k)+"//Process"+str(k)+"//letter"+str(chr(j+97))+"//"
		os.chdir(source_files)
		aux = destination_folder+ str((j+97))+ "//"
		for file in glob.glob('*.jpg'):
			ocorrencias[j] = ocorrencias[j]+1
			os.rename(source_files+file,destination_folder+str(chr(j+97))+"//exemplo"+str(ocorrencias[j])+".jpg")