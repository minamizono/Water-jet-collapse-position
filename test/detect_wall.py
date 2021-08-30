import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.DataFrame({'one': [1,2,3], 'two': [10,20,30]}, index=['a','b','c'])
df2 = pd.DataFrame({'one': [4], 'three': [400]}, index=['d'])

df = pd.concat([df1,df2])

print(df)

#フォルダ指定
output_folder = (r'C:\Users\pcabe1908\Documents\GitHub\Water-jet-collapse-position\output')

# 画像読み込み
calib_img = cv2.imread(r"C:\Users\pcabe1908\Documents\GitHub\Water-jet-collapse-position\input\calib.bmp")
img_on_line = calib_img.copy()
height, width = calib_img.shape[:2]
blank_img = np.zeros((height, width, 3))
blank_img += 255 #←全ゼロデータに255を足してホワイトにする

# グレースケール
gray = cv2.cvtColor(calib_img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("calendar_mod.png", gray)

## 反転 ネガポジ変換
gray2 = cv2.bitwise_not(gray)
cv2.imwrite("calendar_mod2.png", gray2)
lines = cv2.HoughLinesP(gray2, rho=1, theta=np.pi/360, threshold=50, minLineLength=1000, maxLineGap=5)

####壁面検出プログラム########################
x1_inner_left = 0
x2_inner_left = 0
y1_inner_left = 0
y2_inner_left = 0
x1_inner_right = 200
x2_inner_right = 0
y1_inner_right = 0
x2_inner_right = 0
###画像の上に赤線を引く
#インナー左
for line in lines:
    x1, y1, x2, y2 = line[0]
    if x1 < 100 and x1_inner_left < x1:
        x1_inner_left = x1
        x2_inner_left = x2
        y1_inner_left = y1
        y2_inner_left = y2
# 線を引く
red_lines_img = cv2.line(img_on_line, (x1_inner_left,y1_inner_left), (x2_inner_left,y2_inner_left), (0,0,255), 1)
#インナー右
for line in lines:
    x1, y1, x2, y2 = line[0]
    if x1 > 100 and x1_inner_right > x1:
        x1_inner_right = x1
        x2_inner_right = x2
        y1_inner_right = y1
        y2_inner_right = y2
red_lines_img = cv2.line(img_on_line, (x1_inner_right,y1_inner_right), (x2_inner_right,y2_inner_right), (0,0,255), 1)
cv2.imwrite(output_folder+"/inner_line_on_photo.png", red_lines_img)
        
#白背景に黒線を引く
#インナー左
for line in lines:
    x1, y1, x2, y2 = line[0]
    if x1 < 100 and x1_inner_left < x1:
        x1_inner_left = x1
        x2_inner_left = x2
        y1_inner_left = y1
        y2_inner_left = y2
# 線を引く
black_lines_img = cv2.line(blank_img, (x1_inner_left,y1_inner_left), (x2_inner_left,y2_inner_left), (0,0,0), 1)

#インナー右
for line in lines:
    x1, y1, x2, y2 = line[0]
    if x1 > 100 and x1_inner_right > x1:
        x1_inner_right = x1
        x2_inner_right = x2
        y1_inner_right = y1
        y2_inner_right = y2
black_lines_img = cv2.line(blank_img, (x1_inner_right,y1_inner_right), (x2_inner_right,y2_inner_right), (0,0,0), 1)
cv2.imwrite(output_folder+"/inner_line.png", black_lines_img)


