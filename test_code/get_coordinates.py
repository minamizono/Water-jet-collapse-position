import cv2
import numpy as np
import pandas as pd


def main():
    
    # 画像をグレースケールで読み込む
    gray = cv2.imread(input_image, 0)
    cv2.imwrite('01_gray_src.jpg', gray)

    # 2値化フィルターによる輪郭の強調
    contour = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 5)
    cv2.imwrite('02_contour.jpg', contour)

    # 輪郭の座標を読み取る
    contours, hierarchy = cv2.findContours(contour, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    x_list = []
    y_list = []
    for i in range(0, len(contours)):
        buf_np = contours[i].flatten() # numpyの多重配列になっているため、一旦展開する。
        #print(buf_np)
        for i, elem in enumerate(buf_np):
            if i%2==0:
                x_list.append(elem)
            else:
                y_list.append(elem*(-1))

    # pandasのSeries型へ一旦変換
    x_df = pd.Series(x_list)
    y_df = pd.Series(y_list)
    
    # pandasのDataFrame型へ結合と共に、列名も加えて変換
    DF = pd.concat((x_df.rename(r'#X'), y_df.rename('Y')), axis=1, sort=False)
    print(DF)
    DF.to_csv("03_target_contour.csv", encoding="utf-8", index=False)


if __name__ == '__main__':
    # 画像ファイルのパスを指定
    input_image = (r'C:\Users\pcabe1908\Documents\GitHub\Water-jet-collapse-position\output\jet_surface.png')
    
    main()