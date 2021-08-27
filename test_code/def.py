import cv2
import numpy as np
def main():
    kkk=('12345')
    print(kkk)
    aisatu(kkk)

def aisatu(abc):
    img=cv2.imread(r'C:\Users\pcabe1908\Documents\GitHub\Water-jet-collapse-position\output_fig\calib_img_gray_mirror.bmp')
    cv2.imwrite(abc+'.bmp',img)

main()

