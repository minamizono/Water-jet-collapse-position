import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


##=============== 画像読み込み ==============##
wall = cv2.imread(r'C:\Users\pcabe1908\Documents\GitHub\Water-jet-collapse-position\output_fig\inner_line.png')
jet = cv2.imread(r'C:\Users\pcabe1908\Documents\GitHub\Water-jet-collapse-position\output_fig\jet_surface.png')

##=============== 書き込みフォルダ ==============##
output_folder_graph = (r'C:\Users\pcabe1908\Documents\GitHub\Water-jet-collapse-position\output_graph') #グラフをここに書き出す
output_folder_fig = (r'C:\Users\pcabe1908\Documents\GitHub\Water-jet-collapse-position\output_fig') #図をここに書き出す


#########################
#　正中線の位置取得　#
#########################
height , width = wall.shape[:2]
half_index = np.full((height,int(width/2)),0)
print(half_index)


#########################
#　左のギャップ　#
#########################
##=============== 左座標取得 ==============##
wall_coordinate_left = np.argmin(wall,1)
jet_coordinate_left = np.argmin(jet,1)

##=============== 左の座標0以下を削除してプロットする =================##
left_gap = jet_coordinate_left-wall_coordinate_left
left_gap_df = pd.DataFrame(left_gap,columns=['a','b','c'])
left_gap_df2 = left_gap_df.copy()
for i in left_gap_df2.index:
    if (left_gap_df2.loc[i] <= 0).all():
        left_gap_df2.drop(i, axis=0, inplace=True)
left_gap_df2.plot()
plt.savefig(output_folder_graph+'/left_gap.png')
plt.show()


#======================================#
#　右のギャップ　                       #
#　画像を左右反転してそのまま処理をする   #
#======================================#
##=============== 配列左右反転 ===================##
wall_fliplr = np.fliplr(wall)
jet_fliplr = np.fliplr(jet)

##=============== 右座標取得 ====================##
wall_coordinate_right = np.argmin(wall_fliplr,1)
jet_coordinate_right = np.argmin(jet_fliplr,1)

##==============右の座標0以下を削除してプロットする=====================##
right_gap = jet_coordinate_right-wall_coordinate_right
right_gap_df = pd.DataFrame(right_gap,columns=['a','b','c'])
right_gap_df2 = right_gap_df.copy()
for i in right_gap_df2.index:
    if (right_gap_df2.loc[i] <= 0).all():
        right_gap_df2.drop(i, axis=0, inplace=True)
right_gap_df2.plot()
plt.savefig(output_folder_graph+'/right_gap.png')
plt.show()



#===============================#
#　ジェット幅　　　　　　　　　　　#
#　右の線を元の位置として座標取得　#
#===============================#

