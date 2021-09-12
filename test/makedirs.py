import os
import cv2

inputfile = (r'C:\Users\pcabe1908\Documents\GitHub\Water-jet-collapse-position\input')
img = cv2.imread(inputfile+'\0000001.bmp')
cv2.imshow('nemuiyo',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

os.makedirs(inputfile+'sub', exist_ok=True)