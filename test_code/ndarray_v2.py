import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


##=============== 画像読み込み ==============##
output_folder=(r'C:\Users\pcabe1908\Documents\GitHub\Water-jet-collapse-position\output')
wall = cv2.imread(r'C:\Users\pcabe1908\Documents\GitHub\Water-jet-collapse-position\output\inner_line.png')
jet = cv2.imread(r'C:\Users\pcabe1908\Documents\GitHub\Water-jet-collapse-position\output\jet_surface.png')


##=============== 左座標取得 ==============##
wall_coordinate = np.argmin(wall,1)
wall_coordinate = pd.DataFrame(wall_coordinate,columns=['a','b','c'])
wall_coordinate.plot()
plt.savefig('wall.png')
plt.show()

jet_coordinate = np.argmin(jet,1)
jet_coordinate = pd.DataFrame(jet_coordinate,columns=['a','b','c'])
jet_coordinate.plot()
plt.savefig('jet.png')
plt.show()

##=============== 座標０以下を削除してプロットする =================##
left_gap = jet_coordinate-wall_coordinate
left_gap_df = pd.DataFrame(left_gap,columns=['a','b','c'])
left_gap_df2 = left_gap_df.copy()
for c in left_gap_df2.index:
    if (left_gap_df2.loc[c] <= 0).all():
        left_gap_df2.drop(c, axis=0, inplace=True)
print(left_gap_df2)
left_gap_df2.plot()
plt.savefig('left_gap_df2.png')
plt.show()


