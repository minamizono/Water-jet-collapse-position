#======== ライブラリ読み込み ========#
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.reshape.concat import concat

#======== フォルダの指定 ========#
input_folder = (r'C:\Users\pcabe1908\Documents\GitHub\Water-jet-collapse-position\input')                   # 処理する画像ファイルを選択する
output_folder_graph = (r'C:\Users\pcabe1908\Documents\GitHub\Water-jet-collapse-position\output_graph')     # グラフ                を書き出すフォルダを選択する    
output_folder_fig = (r'C:\Users\pcabe1908\Documents\GitHub\Water-jet-collapse-position\output_fig')         # 画像                  を書き出すフォルダを選択する
output_folder_csv = (r'C:\Users\pcabe1908\Documents\GitHub\Water-jet-collapse-position\output_csv')         # csv_基本統計データ     を書き出すフォルダを選択する       
output_folder_csv_original = (output_folder_csv+'\original')                                                # csv_originalデータ    を書き出すフォルダを選択する

#======== グラフパラメータ定義 ============#
plt.rcParams["font.family"] = "Times New Roman"      #全体のフォントを設定
plt.rcParams["xtick.direction"] = "in"               #x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams["ytick.direction"] = "in"               #y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams['axes.grid'] = True                     # make grid
plt.rcParams["xtick.minor.visible"] = True          #x軸補助目盛りの追加
plt.rcParams["ytick.minor.visible"] = True          #y軸補助目盛りの追加
plt.rcParams["xtick.major.width"] = 1.5              #x軸主目盛り線の線幅
plt.rcParams["ytick.major.width"] = 1.5              #y軸主目盛り線の線幅
plt.rcParams["xtick.minor.width"] = 1.0              #x軸補助目盛り線の線幅
plt.rcParams["ytick.minor.width"] = 1.0              #y軸補助目盛り線の線幅
plt.rcParams["xtick.major.size"] = 10                #x軸主目盛り線の長さ
plt.rcParams["ytick.major.size"] = 10                #y軸主目盛り線の長さ
plt.rcParams["xtick.minor.size"] = 5                #x軸補助目盛り線の長さ
plt.rcParams["ytick.minor.size"] = 5                #y軸補助目盛り線の長さ
plt.rcParams["font.size"] = 14                       #フォントの大きさ
plt.rcParams["axes.linewidth"] = 1.5                 #囲みの太さ\

#======== ファイルの読み込み ========#
N = 1000
calib_img =cv2.imread (input_folder+'/calib.bmp') #壁面検出に使用
calib_img_dif = calib_img.copy() #差分をとるのに使用
calib_img_dif = calib_img_dif.astype(np.float32)
calib_img_dif = cv2.cvtColor(calib_img_dif,cv2.COLOR_BGR2GRAY)

#======== pixel換算 =============#
Pixel = 1

#======== 壁面の検出プログラム =========#
#== 白色画像作成 ==#
height, width = calib_img.shape[:2]
blank_img = np.zeros((height, width, 3))
blank_img += 255 #←全ゼロデータに255を足してホワイトにする
blank_img_right_wall = blank_img.copy()
blank_img_left_wall = blank_img.copy()
blank_ig_jet = blank_img.copy()
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
    number_padded = format (i+1,'07')
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
    #cv2.imwrite(output_folder_fig+'/jet_edge_img'+str(number_padded)+'.png', jet_surface) 
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


#y方向の距離をインデックス１に挿入する
y_length = np.arange(1,height+1)
y_length_df = pd.DataFrame(y_length,columns=['jet_break_up_length'])
#print(y_length_df)


#======= 集計した左の隙間がconcat_gapに集計される =======#
#====== 生データ ======#
#== 左のギャップ ==#
concat_gap_left.drop(['A'], axis=1, inplace=True)
concat_gap_left_copy = concat_gap_left.copy()
concat_gap_left.insert(0,'jet_break_up_length',y_length_df)
concat_gap_left.to_csv(output_folder_csv_original+'/1_left_gap.csv')
#concat_gap_left.plot(x ='jet_break_up_length')
#== 右のギャップ ==#
concat_gap_right.drop(['A'], axis=1, inplace=True)
concat_gap_right_copy = concat_gap_right.copy()
concat_gap_right.insert(0,'jet_break_up_length',y_length_df)
concat_gap_right.to_csv(output_folder_csv_original+'/2_right_gap.csv')
#concat_gap_right.plot(x ='jet_break_up_length')
#== ジェットの隙間 ==#
concat_jet.drop(['A'], axis=1, inplace=True)
concat_jet_copy = concat_jet.copy()
concat_jet.insert(0,'jet_break_up_length',y_length_df)
concat_jet.to_csv(output_folder_csv_original+'/3_jet_width.csv')
#concat_jet.plot(x ='jet_break_up_length')
#=====　基本的統計データ =======#
#== 左側の隙間 ==#
concat_gap_left_copy = concat_gap_left_copy.T
concat_gap_left_describe = concat_gap_left_copy.describe().T
concat_gap_left_describe.drop(['count','25%','50%','75%','std'],axis=1,inplace=True)
concat_gap_left_describe.insert(0,'jet_break_up_length',y_length_df)
concat_gap_left_describe = (concat_gap_left_describe.rename(columns={'mean': 'left_gap_mean', 'min': 'left_gap_min','max': 'left_gap_max'}))
concat_gap_left_describe = concat_gap_left_describe * Pixel
concat_gap_left_describe.to_csv(output_folder_csv+'/1_left_describe.csv')
#concat_gap_left_describe.plot(x ='jet_break_up_length')
#== 右側の隙間 ==#
concat_gap_right_copy = concat_gap_right_copy.T
concat_gap_right_describe = concat_gap_right_copy.describe().T
concat_gap_right_describe.drop(['count','25%','50%','75%','std'],axis=1,inplace=True)
concat_gap_right_describe = (concat_gap_right_describe.rename(columns={'mean': 'right_gap_mean', 'min': 'right_gap_min','max': 'right_gap_max'}))
concat_gap_right_describe.insert(0,'jet_break_up_length',y_length_df)
concat_gap_right_describe = concat_gap_right_describe * Pixel
concat_gap_right_describe.to_csv(output_folder_csv+'/2_right_describe.csv')
#concat_gap_right_describe.plot(x ='jet_break_up_length')
#== ジェットの幅 ==#
concat_jet_copy = concat_jet_copy.T
concat_jet_describe = concat_jet_copy.describe().T
concat_jet_describe.drop(['count','25%','50%','75%','std'],axis=1,inplace=True)
concat_jet_describe = (concat_jet_describe.rename(columns={'mean': 'jet_wadth_mean', 'min': 'jet_width_min','max': 'jet_width_max'}))
concat_jet_describe.insert(0,'jet_break_up_length',y_length_df)
concat_jet_describe = concat_jet_describe * Pixel
concat_jet_describe.to_csv(output_folder_csv+'/3_jet_describe.csv')
#concat_jet_describe.plot(x ='jet_break_up_length')
#== 三つのグラフを一つにまとめる ==#
graph_merge = pd.merge(concat_gap_left_describe,concat_gap_right_describe,on='jet_break_up_length')
graph_merge = pd.merge(graph_merge,concat_jet_describe,on='jet_break_up_length')
graph_merge.to_csv(output_folder_csv+'/4_merge_describe.csv')
graph_merge_mean = (graph_merge.iloc[:,[0,1,4,7]]) #meanを取得
#== グラフを描写する_平均値のデータ ==#
graph_merge_mean.plot(x ='jet_break_up_length')
plt.title("jet break up length")
plt.xlabel("jet break up length [mm]")
plt.ylabel("gap [mm]")
plt.xlim(0, height*Pixel)
plt.ylim(0, width*Pixel/2)
plt.savefig(output_folder_graph+'/mean.png')
plt.show()

#== グラフを描写する_最大最少のデータ ==#
graph_merge_min_and_max = (graph_merge.iloc[:,[0,2,5,9]]) #min&maxを取得
graph_merge_min_and_max.plot(x ='jet_break_up_length')
plt.title("jet break up length")
plt.xlabel("jet break up length [mm]")
plt.ylabel("gap [mm]")
plt.xlim(0, height*Pixel)
plt.ylim(0, width*Pixel/2)
plt.savefig(output_folder_graph+'/min_and_max.png')
plt.show()