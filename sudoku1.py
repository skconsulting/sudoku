import cv2
from imutils import contours
import numpy as np
import pytesseract
import math
from skimage.filters import threshold_local
import matplotlib
import matplotlib.pyplot as plt
from pytesseract import Output
from scipy import ndimage
font = cv2.FONT_HERSHEY_SIMPLEX

# image = cv2.imread('3_7.jpg')
# pytesseract.pytesseract.tesseract_cmd = 'C:/Users/sylvain/AppData/Local/Programs/Tesseract-OCR/tesseract.exe'

# boxes = pytesseract.image_to_boxes(image,config='--psm 10 --oem 3 -c tessedit_char_whitelist=123456789') 
# # boxes = pytesseract.image_to_boxes(result,config='--psm 10  -c tessedit_char_whitelist=123456789') 


# print(boxes)
# cv2.imshow('image', image)
#     # cv2.imshow('imageR', imageR)
# cv2.waitKey(175)
        
# ooo
# Load image, grayscale, and adaptive threshold
image = cv2.imread('sudoku1.jpg')
new_width = 500

# dsize
dsize = (new_width, image.shape[0])

# resize image
image = cv2.resize(image, dsize, interpolation = cv2.INTER_AREA)
# img_transf = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
# img_transf[:,:,0] = cv2.equalizeHist(img_transf[:,:,0])
# image = cv2.cvtColor(img_transf, cv2.COLOR_YUV2BGR)

# img_transf = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# clahe = cv2.createCLAHE(tileGridSize=(100,100))
# img_transf[:,:,2] = clahe.apply(img_transf[:,:,2])
# image = cv2.cvtColor(img_transf, cv2.COLOR_HSV2BGR)


# ratio = (int(image.shape[0] / 500.0),500)
# image = cv2.resize(image, ratio)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# gray = cv2.GaussianBlur(gray, (7, 7), 3)
# thresh = cv2.adaptiveThreshold(blurred, 255,
# 		cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
# 	thresh = cv2.bitwise_not(thresh)
thresh = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 57, 5)

# Filter out all numbers and noise to isolate only boxes
cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    area = cv2.contourArea(c)
    if area < 1000:
        cv2.drawContours(thresh, [c], -1, (0, 0, 0), -1)
# thresh1=thresh.copy()


# Fix horizontal and vertical lines
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
                          vertical_kernel, iterations=9)
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
                          horizontal_kernel, iterations=4)
# cv2.imshow('gray', gray)
# cv2.imshow('thresh', thresh)
# # cv2.imshow('thresh1', thresh1)


# Sort by top to bottom and each row by left to right
invert = 255 - thresh
cnts = cv2.findContours(invert, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
(cnts, _) = contours.sort_contours(cnts, method="top-to-bottom")

sudoku_rows = []
row = []
for (i, c) in enumerate(cnts, 1):
    area = cv2.contourArea(c)
    if area < 50000:
        row.append(c)
        if i % 9 == 0:
            (cnts, _) = contours.sort_contours(row, method="left-to-right")
            sudoku_rows.append(cnts)
            row = []

# Iterate through each box
# cv2.imshow('invert', invert)
# cv2.waitKey()
list_images=[]

imageR = np.zeros(gray.shape, dtype=np.uint8)
imageR2 = np.zeros(gray.shape, dtype=np.uint8)
grid = np.zeros(image.shape, dtype=np.uint8)


tabresr= np.zeros((9,9), dtype=np.uint8)
# thresh=255-thresh

kernel = np.ones((5,5),np.uint8)
thresh = cv2.erode(thresh,kernel,iterations = 1)
kernel = np.ones((3,3),np.uint8)
thresh = cv2.dilate(thresh,kernel,iterations = 1)
thresh=255-thresh

label, num_label = ndimage.label(thresh == 255)
# print(label)

size = np.bincount(label.ravel().astype(int))
biggest_label = size[1:].argmax() + 1
clump_mask = label == biggest_label
imageR2[label==biggest_label]=255
# print(biggest_label)


# print(thresh.shape)
# cv2.imshow('imageR2', imageR2)
y,x =np.where(imageR2>254)
# print(i)
# print(np.unique(imageR2))
# ooo
minx = min(x)
maxx=max(x)
miny = min(y)
maxy=max(y)
dx=maxx-minx
dy=maxy-miny
# maxii=maxi
# max1=maxj
dx9=dx/9
dy9=dy/9
print(minx,maxx,miny,maxy,dx,dy,dx/9,dy/9)
# print(imageR2.shape)
# print(grid.shape)
tabzoneh={}
tabzonev={}
 
for i in range(9):
    x=minx+(dx9*(i))
    y=x+dx9
    tabzoneh[i]=(x,y)
    
for i in range(9):
    x=miny+(dy9*(i))
    y=x+dy9
    # print(x,y)
    tabzonev[i]=(x,y)
# ooo
for x in range(0,9):
   for y in range(0,9):
        x0=int(tabzoneh[x][0])
        x1=int(tabzoneh[x][1])
        y0=int(tabzonev[y][0])
        y1=int(tabzonev[y][1])
        # print(x0,x1,y0,y1)
        
        px =int((tabzoneh[x][0]+tabzoneh[x][1])/2)
        py=int((tabzonev[y][0]+tabzonev[y][1])/2)
        cv2.putText(grid, str(x)+','+str(y),(px,py), font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        grid=cv2.rectangle(grid, (x1,y1), (x0,y0), (255,0,0), 2)

        if x==0 and y==0:
            px =minx
            py =miny
            cv2.putText(grid, str(x)+','+str(y),(px,py), font, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
        if x==0 and y==8:
            px =minx
            py =maxy
            cv2.putText(grid, str(x)+','+str(y),(px,py), font, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
        if x==8 and y==0:
            px =maxx
            py =miny
            cv2.putText(grid, str(x)+','+str(y),(px,py), font, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
    
        if x ==8 and y ==8:
            px =maxx
            py =maxy
            cv2.putText(grid, str(x)+','+str(y),(px,py), font, 0.5, (255, 0, 255), 1, cv2.LINE_AA)

imageR2color= cv2.cvtColor(imageR2, cv2.COLOR_GRAY2RGB)
# print(imageR2color.shape,imageR2color.min(),imageR2color.max())
# print(grid.shape)
grid =cv2.addWeighted(grid,1,imageR2color,0.1,0)
# cv2.imshow('image', image)
# cv2.imshow('imageR2', imageR2)
# cv2.imshow('imageR2color', imageR2color)
# cv2.imshow('grid', grid)
# cv2.waitKey()
# ooo

for nrow,row in enumerate(sudoku_rows):
    for nc,c in enumerate(row):
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)
        result = cv2.bitwise_and(gray, mask)
        # custom_config = r'--oem 3 --psm 6'
        # img = 255-result
        # list_images.append(pytesseract.image_to_string(img, config=custom_config))
        # h, w, c = result.shape
        # boxes = pytesseract.image_to_boxes(result) 
        # for b in boxes.splitlines():
        #     b = b.split(' ')
        #     result = cv2.rectangle(result, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)
        
        # cv2.imshow('img', result)
        # cv2.waitKey(0)
        # contour, hier = cv2.findContours (result,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        # cnts = sorted(contour, key=cv2.contourArea,reverse=True)[:1]
        # for cnt in cnts:
        #     rect = cv2.boundingRect(cnt)
        #     #cv2.rectangle(c2, (rect[0],rect[1]), (rect[2]+rect[0],rect[3]+rect[1]), (255,255,255), 2)
        #     c2 = result[rect[1]:rect[3]+rect[1],rect[0]:rect[2]+rect[0]]
        #     c2= cv2.copyMakeBorder(c2,5,5,5,5,cv2.BORDER_CONSTANT,value=(0,0,0))
        #     list_images.append(c2)
        # # print(image_to_num(result))
        # image_to_num(result)
        result[mask == 0] = 255
        result2=255-result
        # result2[result2<50]=0
        # result2==np.where(result2<255,0,result2)
        
        # cv2.imshow('result2', result2)
        # cv2.imshow('image', result)
        # cv2.imshow('mask', mask)
        # cv2.waitKey(175)
        # cv2.waitKey()
        imageR[result2>100]=255
        # result=255-result

        np.putmask(result,result > 128,255)
        np.putmask(result,result < 129,0)
        # cv2.imwrite(str(nc)+'_'+str(nrow)+'.jpg',result)
        # cv2.imshow('imagea', result)
        
        # kernel = np.ones((3,3),np.uint8)
        # result = cv2.dilate(result,kernel,iterations = 1)
        # kernel = np.ones((4,4),np.uint8)
        # result = cv2.erode(result,kernel,iterations = 1)
        
        # kernel = np.ones((3,3),np.uint8)
        # result = cv2.dilate(result,kernel,iterations = 1)
     
        # thresh=255-thresh


        # cv2.imshow('result2', result2)
        # cv2.imshow('image', result)
        # # cv2.imshow('imageR', imageR)
        # cv2.waitKey()
        
        # cv2.waitKey()
        # imageR+=result
        # img = 255-result
        # text = pytesseract.image_to_string(result, lang="eng",config='--psm 6 --oem 3')
        # boxes = pytesseract.image_to_boxes(result,lang="eng",config='--psm 6 --oem 3') 
        # boxes = pytesseract.image_to_boxes(result,config='--psm 6 --oem 3') 
        boxes = pytesseract.image_to_boxes(result,config='--psm 10 --oem 3 -c tessedit_char_whitelist=123456789') 
        # boxes = pytesseract.image_to_boxes(result,config='--psm 10  -c tessedit_char_whitelist=123456789') 



        boxes=boxes.split(' ')
        # # print(boxes,boxes[0])
        
        try:
            restext=int(boxes[0])
        except:
            # print(boxes)
            restext=0
        # # print(restext)
        # # print(text[0])
        if restext in [1,2,3,4,5,6,7,8,9]:
        # if restext in [2]:
            # print(boxes[1])
            # print(np.unique(result))

            x0=int(boxes[1])
            y0=maxx-int(boxes[2])+minx
            x1=int(boxes[3])
            y1=maxx-int(boxes[4])+minx
            mx=int((x0+x1)/2)
            my=int((y0+y1)/2)
            # if restext==2 and recToDraw:
            if True:
                imageR=cv2.rectangle(imageR, (x0,y0), (x1,y1), (255,0,0), 2)
                cv2.putText(imageR,str(restext),(mx,my), font, 1, (255, 255, 255), 1, cv2.LINE_AA)


                print(restext,x0,x1,y0,y1,nrow,nc)
                print(mx,my)
                print(boxes)
             
               
                recToDraw=False
            ir=0
            jr=0    
            for i in range(9):
                # x0=int(tabzoneh[x][0])
                # x1=int(tabzoneh[x][1])
                # y0=int(tabzonev[y][0])
                # y1=int(tabzonev[y][1])
                if mx> tabzoneh[i][0] and mx < tabzoneh[i][1]:
                    ir=i
                    break
            for i in range(9):
                if my> tabzonev[i][0] and my < tabzonev[i][1]:
                    jr=i
                    break
            tabresr[jr,ir]=restext
        # if result.min() <100:
        #     cv2.imshow('image', result)
        #     print(result.min())
        #     print(boxes,nrow,nc)
        #     cv2.waitKey()
                    
                    
        # else:
        #     restext =0
        
        # tabresr[nrow,nc]=restext
        # list_images.append(text[0])
        # list_images.append(pytesseract.image_to_string(result))

        # d=pytesseract.image_to_data(result, output_type=Output.DICT)
    
        # cv2.imshow('result', result)
        # result2=255-result
        # result2[result2<50]=0
        # result2==np.where(result2<255,0,result2)
        
        # cv2.imshow('result2', result2)
        # cv2.imshow('image', result)
        # cv2.imshow('mask', mask)
        # cv2.waitKey(175)
        # cv2.waitKey()

        # imageR[result2>100]=255
        # result[mask == 0] = 255
        # imageR+=result
        # cv2.imshow('result', result)
        # cv2.imshow('imageR', imageR)
        # cv2.waitKey(175)
        # cv2.waitKey()

# cv2.imshow('thresh', thresh)
# print(d.keys())

# print(list_images)
# print(tabresr)
# d=pytesseract.image_to_data(imageR, output_type=Output.DICT,config=custom_config)
# print(d)
# print(pytesseract.image_to_string(imageR, config=custom_config))
# cv2.imshow('imageR', imageR)
# cv2.imshow('gray', gray)
# thresh=255-thresh

# kernel = np.ones((5,5),np.uint8)
# thresh = cv2.erode(thresh,kernel,iterations = 1)
# kernel = np.ones((3,3),np.uint8)
# thresh = cv2.dilate(thresh,kernel,iterations = 1)
# thresh=255-thresh

# from skimage import measure

# np.putmask(pred,predorig > th,255)

# predm = measure.label(thresh)

# for region in measure.regionprops(predm):
#     # retrieve x, y, height and width
#     y, x, y2, x2 = region.bbox
#     height = y2 - y
#     width = x2 - x
#     print(x,y,x2,y2)


    
    
print(tabresr)
# print(tabzonev)
    
cv2.imshow('imageR', imageR)
cv2.imshow('gray', gray)

cv2.waitKey()
cv2.destroyAllWindows()  
