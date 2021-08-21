import cv2
import numpy as np
#画像の読み込み
pic1=cv2.imread(r'C:\code\jet\Water-jet-collapse-position\calib.bmp',cv2.IMREAD_GRAYSCALE)
pic1=pic1.astype(np.float32)
pic2=cv2.imread(r'C:\code\jet\Water-jet-collapse-position\0000026.bmp',cv2.IMREAD_GRAYSCALE)
pic2=pic2.astype(np.float32)
#輝度差し引き
pic3=pic1-pic2
pic3=np.where(pic3<0,0,pic3)
pic3=pic3.astype(np.uint8)

pic4=cv2.imread(r'C:\code\jet\Water-jet-collapse-position\muji.jpg')

print(type(pic1))
print(type(pic2))
print(type(pic3))
print(type(pic4))
#画像出力
cv2.imwrite('pic3.jpg', pic3)

ret, img_binary = cv2.threshold(pic3, 100, 255,cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#contours = list(filter(lambda x: cv2.contourArea(x) > 25000, contours))
max_contour = max(contours, key=lambda x: cv2.contourArea(x))


img_contour = cv2.drawContours(pic4, max_contour, -1, (000, 000, 000), 1)

cv2.imshow("edge",cv2.cvtColor(img_contour, cv2.COLOR_BGR2RGB))
cv2.waitKey()
cv2.destroyAllWindows()

cv2.imwrite('jet.edge.jpg',img_contour)