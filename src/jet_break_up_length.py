#======== ライブラリ読み込み ========#
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#======== フォルダの指定 ========#
input_folder = (r'C:\Users\pcabe1908\Documents\GitHub\Water-jet-collapse-position\input')
output_folder_graph = (r'C:\Users\pcabe1908\Documents\GitHub\Water-jet-collapse-position\output_graph')
output_folder_fig = (r'C:\Users\pcabe1908\Documents\GitHub\Water-jet-collapse-position\output_fig')

#======== ファイルの読み込み ========#
N = 2
calib_img =cv2.imread (input_folder+'/calib.bmp')

#======== 壁面の検出プログラム =========#
#== 白色画像作成 ==#
img_on_line = calib_img.copy()
height, width = calib_img.shape[:2]
blank_img = np.zeros((height, width, 3))
blank_img += 255 #←全ゼロデータに255を足してホワイトにする
blank_img_right = blank_img.copy()
blank_img_left = blank_img.copy()
#== グレースケール化 ==#
calib_img_gray = cv2.cvtColor(calib_img,cv2.COLOR_BGR2GRAY)
cv2.imwrite(output_folder_fig+'/calib_img_gray.bmp',calib_img_gray)
#== ネガポジ反転 ==#
calib_img_gray_nega = cv2.bitwise_not(calib_img_gray)
cv2.imwrite(output_folder_fig+'/calib_img_gray_nega.bmp',calib_img_gray_nega)
#== ハフ変換で直線検出 ==#
calib_img_gray_nega_lines = cv2.HoughLinesP(calib_img_gray_nega, rho=1, theta=np.pi/360, threshold=50, minLineLength=1000, maxLineGap=5)
#== 壁面検出プログラムの初期値 ==#
x1_inner_left = 0
x2_inner_left = 0
y1_inner_left = 0
y2_inner_left = 0
x1_inner_right = 200
x2_inner_right = 0
y1_inner_right = 0
y2_inner_right = 0
center_line=100
#== インナー右 ==#
for line in calib_img_gray_nega_lines:
    x1, y1, x2, y2 = line[0]
    if x1 > center_line and x1_inner_right > x1:
        x1_inner_right = x1
        x2_inner_right = x2
        y1_inner_right = y1
        y2_inner_right = y2
inner_line = cv2.line(calib_img, (x1_inner_right,y1_inner_right), (x2_inner_right,y2_inner_right), (0,0,255), 3)
blank_img_right = cv2.line(blank_img_right, (x1_inner_right,y1_inner_right), (x2_inner_right,y2_inner_right), (0,0,0), 1)
#== 初期値リセット ==#
x1_inner_left = 0
x2_inner_left = 0
y1_inner_left = 0
y2_inner_left = 0
x1_inner_right = 200
x2_inner_right = 0
y1_inner_right = 0
y2_inner_right = 0
center_line=100
#== インナー左　==#
for line in calib_img_gray_nega_lines:
    x1, y1, x2, y2 = line[0]
    if x1 < center_line and x1_inner_left < x1:
        x1_inner_left = x1
        x2_inner_left = x2
        y1_inner_left = y1
        y2_inner_left = y2
inner_line = cv2.line(calib_img, (x1_inner_left,y1_inner_left), (x2_inner_left,y2_inner_left), (0,0,255), 3)
blank_img_left = cv2.line(blank_img_left, (x1_inner_left,y1_inner_left), (x2_inner_left,y2_inner_left), (0,0,000), 1)
#======== 検出壁面を描写 =========#
cv2.imwrite(output_folder_fig+'/calib_img_lines.bmp', inner_line)
cv2.imwrite(output_folder_fig+'/inner_right_line.bmp', blank_img_right)
cv2.imwrite(output_folder_fig+'/inner_left_line.bmp', blank_img_left)
#======== 座標保持 ============#
#== 左側の壁の座標を取得 ==#
wall_coordinate_left = np.argmin(blank_img_left,1)
df_wall_coordinate_left = pd.DataFrame(wall_coordinate_left,columns=['a1','b1','c1'])
#== 右側の壁の座標を取得 ==#
wall_coordinate_right = np.argmin(blank_img_right,1)
df_wall_coordinate_right = pd.DataFrame(wall_coordinate_right,columns=['a2','b2','c2'])
#== 縮小拡大管内の幅を検出 ==#
wall_gap = wall_coordinate_right - wall_coordinate_left
df_wall_gap = pd.DataFrame(wall_gap,columns=['a3','b3','c3'])
#== 列を削除 ==#
df_wall_coordinate_left.drop(['b1', 'c1'], axis=1, inplace=True)
df_wall_coordinate_right.drop(['b2', 'c2'], axis=1, inplace=True)
df_wall_gap.drop(['b3', 'c3'], axis=1, inplace=True)

#======== 座標


#======== バッチ処理 =========#
#for i in range(N):
    #print('current batch is ; '+str(i+1)+'/'+str(N))
    #number_padded = format (i+1,'03')

