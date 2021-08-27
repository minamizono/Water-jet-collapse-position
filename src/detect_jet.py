import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#読見込む画像フォルダ
input_folder = (r'C:\Users\pcabe1908\Documents\GitHub\Water-jet-collapse-position\input')
#書き出す画像フォルダ
output_folder = (r'C:\Users\pcabe1908\Documents\GitHub\Water-jet-collapse-position\output')

#画像の読み込み
calib_img=cv2.imread(r'C:\Users\pcabe1908\Documents\GitHub\Water-jet-collapse-position\input\calib.bmp',cv2.IMREAD_GRAYSCALE)
calib_img=calib_img.astype(np.float32)
jet_img=cv2.imread(r'C:\Users\pcabe1908\Documents\GitHub\Water-jet-collapse-position\input\001.bmp',cv2.IMREAD_GRAYSCALE)
jet_img=jet_img.astype(np.float32)
#輝度差し引き
diff_img=calib_img-jet_img
diff_img=np.where(diff_img<0,0,diff_img)
diff_img=diff_img.astype(np.uint8)

#calib_imgと同サイズ；無地の白色画像を作成
height, width = calib_img.shape[:2]
blank_img = np.zeros((height, width, 3))
blank_img += 255 #←全ゼロデータに255を足してホワイトにする

###差分画像出力
###cv2.imwrite('diff_img.jpg', diff_img)

#二値化画像を作成
ret, img_binary = cv2.threshold(diff_img, 100, 255,cv2.THRESH_BINARY)

#輪郭抽出
contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

#フィルター作成
#contours = list(filter(lambda x: cv2.contourArea(x) > 25000, contours))

#一番大きな輪郭を抽出する
max_contour = max(contours, key=lambda x: cv2.contourArea(x))


#白色画像に最大輪郭を描きだす
img_contour = cv2.drawContours(blank_img, max_contour, -1, (000, 000, 000), 1)
#np.set_printoptions(threshold=np.inf)
#print(np.argmax(max_contour,2))

img_argmin_left = np.argmin(img_contour,1)
#print(img_argmin_left)

df = pd.DataFrame(img_argmin_left,columns=['a','b','c'])
del df['b']
del df['c']
print(df)


    #img_argmin_right = np.fliplr(img)
    #cv2.imwrite(output_folder+'/test_after_right.png',img_argmin_right)
    #img_argmin_right = np.argmin(img_argmin_right,1)
    #img_argmin_right = cv2.flip(img_argmin_right, 1)
    #print(img_argmin_right)
df.plot()
plt.show()

cv2.imwrite(output_folder+'/jet_surface.png',img_contour)
