#======== ライブラリ読み込み ========#
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.reshape.concat import concat

#======== フォルダの指定 ========#
input_folder = (r'C:\Users\pcabe1908\Documents\GitHub\Water-jet-collapse-position\input')
output_folder_graph = (r'C:\Users\pcabe1908\Documents\GitHub\Water-jet-collapse-position\output_graph')
output_folder_fig = (r'C:\Users\pcabe1908\Documents\GitHub\Water-jet-collapse-position\output_fig')
output_folder_csv = (r'C:\Users\pcabe1908\Documents\GitHub\Water-jet-collapse-position\output_csv')

#======== ファイルの読み込み ========#
N = 2
calib_img =cv2.imread (input_folder+'/calib.bmp') #壁面検出に使用
calib_img_dif = calib_img.copy() #差分をとるのに使用
calib_img_dif = calib_img_dif.astype(np.float32)
calib_img_dif = cv2.cvtColor(calib_img_dif,cv2.COLOR_BGR2GRAY)



#======== 壁面の検出プログラム =========#
#== 白色画像作成 ==#
height, width = calib_img.shape[:2]
blank_img = np.zeros((height, width, 3))
blank_img += 255 #←全ゼロデータに255を足してホワイトにする
blank_img_right_wall = blank_img.copy()
blank_img_left_wall = blank_img.copy()
blank_img_jet = blank_img.copy()
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
center_line = width/2
#== インナー右 ==#
for line in calib_img_gray_nega_lines:
    x1, y1, x2, y2 = line[0]
    if x1 > center_line and x1_inner_right > x1:
        x1_inner_right = x1
        x2_inner_right = x2
        y1_inner_right = y1
        y2_inner_right = y2
inner_line = cv2.line(calib_img, (x1_inner_right,y1_inner_right), (x2_inner_right,y2_inner_right), (0,0,255), 3)
blank_img_right_wall = cv2.line(blank_img_right_wall, (x1_inner_right,y1_inner_right), (x2_inner_right,y2_inner_right), (0,0,0), 1)
#== 初期値リセット ==#
x1_inner_left = 0
x2_inner_left = 0
y1_inner_left = 0
y2_inner_left = 0
x1_inner_right = 200
x2_inner_right = 0
y1_inner_right = 0
y2_inner_right = 0
center_line = width/2
#== インナー左　==#
for line in calib_img_gray_nega_lines:
    x1, y1, x2, y2 = line[0]
    if x1 < center_line and x1_inner_left < x1:
        x1_inner_left = x1
        x2_inner_left = x2
        y1_inner_left = y1
        y2_inner_left = y2
inner_line = cv2.line(calib_img, (x1_inner_left,y1_inner_left), (x2_inner_left,y2_inner_left), (0,0,255), 3)
blank_img_left_wall = cv2.line(blank_img_left_wall, (x1_inner_left,y1_inner_left), (x2_inner_left,y2_inner_left), (0,0,000), 1)
#======== 検出壁面を描写 =========#
cv2.imwrite(output_folder_fig+'/calib_img_lines.bmp', inner_line)
cv2.imwrite(output_folder_fig+'/inner_right_line.bmp', blank_img_right_wall)
cv2.imwrite(output_folder_fig+'/inner_left_line.bmp', blank_img_left_wall)
#======== 座標保持 ============#
#== 左側の壁の座標を取得 ==#
wall_coordinate_left = np.argmin(blank_img_left_wall,1)
df_wall_coordinate_left = pd.DataFrame(wall_coordinate_left,columns=['left_wall','b1','c1'])
#== 右側の壁の座標を取得 ==#
wall_coordinate_right = np.argmin(blank_img_right_wall,1)
df_wall_coordinate_right = pd.DataFrame(wall_coordinate_right,columns=['right_wall','b2','c2'])
#== 縮小拡大管内の幅を検出 ==#
wall_gap = wall_coordinate_right - wall_coordinate_left
df_wall_gap = pd.DataFrame(wall_gap,columns=['SI_gap','b3','c3'])
#== 列を削除 ==#
df_wall_coordinate_left.drop(['b1', 'c1'], axis=1, inplace=True)
df_wall_coordinate_right.drop(['b2', 'c2'], axis=1, inplace=True)
df_wall_gap.drop(['b3', 'c3'], axis=1, inplace=True)
#======== 座標をグラフにプロットする ==========#
#df_wall_gap.plot()
#plt.savefig(output_folder_graph+'/SI_wall_gap.png')
#plt.show()


#======== 噴流崩壊の検出プログラム =========#
#== DateFrameの入れ物を作成 ==#
concat_gap_empty_left = np.zeros_like(wall_coordinate_left)
concat_gap_left = pd.DataFrame(concat_gap_empty_left,columns=['A','B','C'])
concat_gap_left.drop(['B', 'C'], axis=1, inplace=True)

concat_gap_empty_right = np.zeros_like(wall_coordinate_left)
concat_gap_right = pd.DataFrame(concat_gap_empty_right,columns=['A','B','C'])
concat_gap_right.drop(['B', 'C'], axis=1, inplace=True)

concat_jet_empty = np.zeros_like(wall_coordinate_left)
concat_jet = pd.DataFrame(concat_jet_empty,columns=['A','B','C'])
concat_jet.drop(['B', 'C'], axis=1, inplace=True)
##== SI壁面左右反転して配列取得 ==##
wall_fliplr = np.fliplr(blank_img_right_wall)
wall_fliplr_coordinate = np.argmin(wall_fliplr,1)
#======== バッチ処理 =========#
#== 画像の読み込み ==#
for i in range(N):
    print('current batch is ; '+str(i+1)+'/'+str(N))
    number_padded = format (i+1,'03')
    img_val = cv2.imread(input_folder+'/'+str(number_padded)+'.bmp',cv2.IMREAD_GRAYSCALE)
    img_val = img_val.astype(np.float32)
    diff_img=calib_img_dif-img_val
    diff_img=np.where(diff_img<0,0,diff_img)
    diff_img=diff_img.astype(np.uint8)
    #== 差分画像書き出し=================================================#
    #==cv2.imwrite('diff_img'+str(number_padded)+'.jpg', diff_img) =====#
    #== 輪郭抽出 ==#
    ret, img_binary = cv2.threshold(diff_img, 100, 255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_contour = max(contours, key=lambda x: cv2.contourArea(x))
    jet_surface = cv2.drawContours(blank_img, max_contour, -1, (000, 000, 000), 1)
    #== blank_img一枚に上書きされないようにリセットする ===================#
    blank_img = np.zeros((height, width, 3))
    blank_img += 255 #←全ゼロデータに255を足してホワイトにする
    #== 輪郭を白色の紙に描写 ======================#
    cv2.imwrite(output_folder_fig+'/jet_edge_img'+str(number_padded)+'.png', jet_surface) 
    ######################
    # 座標取得 
    ######################
    ##====== 左側の座標取得 =======##
    jet_coordinate_left =np.argmin(jet_surface,1)
    ##=============== 左の座標0以下を削除してプロットする =================##
    left_gap = jet_coordinate_left - wall_coordinate_left
    left_gap_df = pd.DataFrame(left_gap,columns=['left_gap_'+str(number_padded),'b'+str(number_padded),'c'+str(number_padded)])
    left_gap_df.drop(['b'+str(number_padded), 'c'+str(number_padded)], axis=1, inplace=True)
    left_gap_df2 = left_gap_df.copy()
    for i in left_gap_df2.index:
        if (left_gap_df2.loc[i] <= 0).all():
            left_gap_df2.drop(i, axis=0, inplace=True)

    ##====== 右側の座標取得 (画像を軸対象に反転させ取得) =======##
    ##== ジェットを左右配列反転 ==#
    jet_surface_fliplr = np.fliplr(jet_surface)
    jet_surface_fliplr_coordinate = np.argmin(jet_surface_fliplr,1)
    ##== 右のギャップを取得（0以下は削除してプロット） ==#
    right_gap = jet_surface_fliplr_coordinate - wall_fliplr_coordinate
    right_gap_df = pd.DataFrame(right_gap,columns=['rigfht_gap'+str(number_padded),'b'+str(number_padded),'c'+str(number_padded)])
    right_gap_df.drop(['b'+str(number_padded), 'c'+str(number_padded)], axis=1, inplace=True)
    right_gap_df2 = right_gap_df.copy()
    for i in right_gap_df2.index:
        if (right_gap_df2.loc[i] <= 0).all():
            right_gap_df2.drop(i, axis=0, inplace=True)
    
    #== ジェットの太さを検出　==#
    jet_width = wall_gap - left_gap - right_gap
    jet_width_df = pd.DataFrame(jet_width,columns=['jet_width'+str(number_padded),'b'+str(number_padded),'c'+str(number_padded)])
    jet_width_df.drop(['b'+str(number_padded), 'c'+str(number_padded)], axis=1, inplace=True)
    jet_width_df2 = jet_width_df.copy()
    for i in jet_width_df2.index:
        if (jet_width_df2.loc[i] >= width/5*4).all(): #大きい数字をカットする
            jet_width_df2.drop(i, axis=0, inplace=True)


    #== 各写真のdateframeを一つにまとめる　==#  
    concat_gap_left = pd.concat([concat_gap_left,left_gap_df2],axis=1)
    concat_gap_right = pd.concat([concat_gap_right,right_gap_df2],axis=1)
    concat_jet = pd.concat([concat_jet,jet_width_df2],axis=1)

    #############################################################
    #==各ループの画像をプロットグラフpng保存したければこれ　==#
    #left_gap_df2.plot()
    #plt.savefig(output_folder_graph+'/left_gap'+str(number_padded)+'.png')    
    #right_gap_df2.plot()
    #plt.savefig(output_folder_graph+'/right_gap'+str(number_padded)+'.png')
    #jet_width_df2.plot()
    #plt.savefig(output_folder_graph+'/jet_width'+str(number_padded)+'.png')
    #plt.show()
    ###########################################################


#======= 集計した左の隙間がconcat_gapに集計される =======#
#== 左のギャップ ==#
concat_gap_left.drop(['A'], axis=1, inplace=True)
concat_gap_left.to_csv(output_folder_csv+'/test_left.csv')
concat_gap_left.plot()
#== 右のギャップ ==#
concat_gap_right.drop(['A'], axis=1, inplace=True)
concat_gap_right.to_csv(output_folder_csv+'/test_right.csv')
concat_gap_right.plot()
#== ジェットの隙間 ==#
concat_jet.drop(['A'], axis=1, inplace=True)
concat_jet.to_csv(output_folder_csv+'/test_jet.csv')
concat_jet.plot()

plt.show()
