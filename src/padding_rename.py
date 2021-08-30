import glob
import os

#連番でリネームするフォルダを選択する
file = glob.glob(r"D:\3_研究室\2_コード\koyama\original\*.bmp") 

for e,i in enumerate(file):
    name, ext = os.path.splitext(i)
    z = e+1
    title = name.rsplit('\\',1)[0]
    os.rename(i,title + '\\{0:04d}'.format(z) + ext) #パディングするのを決める[0:04←こいつが桁数を決めている]


print('finish')