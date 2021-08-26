import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

output_folder=(r'C:\Users\pcabe1908\Documents\GitHub\Water-jet-collapse-position\output')
img = cv2.imread(r'C:\Users\pcabe1908\Documents\GitHub\Water-jet-collapse-position\input\test.png')


img_argmin_left = np.argmin(img,1)
#print(img_argmin_left)

df = pd.DataFrame(img_argmin_left,columns=['a','b','c'])
del df['b']
del df['c']
print(df)


    #img_argmin_right = np.fliplr(img)
    #cv2.imwrite(output_folder+'/test_after_right.png',img_argmin_right)
    #img_argmin_right = np.argmin(img_argmin_right,1)
    #img_argmin_right = cv2.flip(img_argmin_right, 1)
    #print(img_argmin_right)
df.plot()
plt.show()
df.to_csv('test.csv')
cv2.imwrite(output_folder+'/test_after.png',img)
cv2.imshow('test',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


