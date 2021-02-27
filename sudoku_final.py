# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 18:20:38 2021

@author: sylvain
"""

# USAGE
# python adaptive_equalization.py --image images/boston.png


from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import numpy as np
import imutils
import cv2
import pytesseract
import os
from appJar import gui

font = cv2.FONT_HERSHEY_SIMPLEX

debug=True
solve=False

finalwidth=600
imageToLoad='sudoku1.jpg'


# image = cv2.imread('s1.jpg')
# image = cv2.imread('s2.png')
# image = cv2.imread('s1.jpg')
# image = cv2.imread('sexpert.jpg')


def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

# load the input image from disk and convert it to grayscale


# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# gray = cv2.GaussianBlur(gray, (7, 7), 3)
# image = cv2.adaptiveThreshold(gray, 255,
#     		cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 57, 2)


# image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)



# cv2.imshow("image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# ooo

def find_puzzle(image, debug=False):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)
    # apply adaptive thresholding and then invert the threshold map
    thresh = cv2.adaptiveThreshold(blurred, 255,
    		cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)
    if debug:
        cv2.imshow("Puzzle Thresh", thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
# find contours in the thresholded image and sort them by size in
	# descending order
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    		cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    	# initialize a contour that corresponds to the puzzle outline
    puzzleCnt = None
    	# loop over the contours
    for c in cnts:
    		# approximate the contour
    		peri = cv2.arcLength(c, True)
    		approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    		# if our approximated contour has four points, then we can
    		# assume we have found the outline of the puzzle
    		if len(approx) == 4:
    			puzzleCnt = approx
    			break
    if puzzleCnt is None:
    		raise Exception(("Could not find Sudoku puzzle outline. "
    			"Try debugging your thresholding and contour steps."))
    if debug:
    		# draw the contour of the puzzle on the image and then display
    		# it to our screen for visualization/debugging purposes
        output = image.copy()
        cv2.drawContours(output, [puzzleCnt], -1, (0, 255, 0), 2)
        cv2.imshow("Puzzle Outline", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
# apply a four point perspective transform to both the original
	# image and grayscale image to obtain a top-down bird's eye view
	# of the puzzle
    puzzle = four_point_transform(image, puzzleCnt.reshape(4, 2))
    warped = four_point_transform(gray, puzzleCnt.reshape(4, 2))
	# check to see if we are visualizing the perspective transform
    if debug:
    		# show the output warped image (again, for debugging purposes)
            cv2.imshow("Puzzle Transform", puzzle)
            cv2.imshow("warped", warped)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    return (puzzle, warped)


def extract_digit(cell, debug=False):
	# apply automatic thresholding to the cell and then clear any
	# connected borders that touch the border of the cell
    thresh = cv2.threshold(cell, 0, 255,
		cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)
	# check to see if we are visualizing the cell thresholding step
    if debug:
        cv2.imshow("Cell Thresh", thresh)
        cv2.waitKey(0)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
	# if no contours were found than this is an empty cell
    if len(cnts) == 0:
        return None
	# otherwise, find the largest contour in the cell and create a
	# mask for the contour
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)
    # compute the percentage of masked pixels relative to the total
	# area of the image
    (h, w) = thresh.shape
    percentFilled = cv2.countNonZero(mask) / float(w * h)
	# if less than 3% of the mask is filled then we are looking at
	# noise and can safely ignore the contour
    if percentFilled < 0.03:
        # print(percentFilled)
        # cv2.imshow("mask", mask)
        # # cv2.imshow("Digit", roi)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return None
	# apply the mask to the thresholded cell
    digit = cv2.bitwise_and(thresh, thresh, mask=mask)
	# check to see if we should visualize the masking step
    if debug:
        cv2.imshow("Digit", digit)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
	# return the digit to the calling function
    return digit

def click_and_crop(event, x, y, flags, param):
    global quitl,pattern,dirpath_patient

    if event == cv2.EVENT_LBUTTONDOWN:
        # cv2.rectangle(menus, (150,12), (370,32), black, -1)
        # posrc=0
        print (x,y)
#         for key1 in classif:
#             labelfound=False
#             xr=5
#             yr=15*posrc
#             xrn=xr+10
#             yrn=yr+10
#             if x>xr and x<xrn and y>yr and y< yrn:

#                 print ('this is',key1)
#                 pattern=key1
#                 cv2.rectangle(menus, (200,0), (210,10), classifc[pattern], -1)
#                 cv2.rectangle(menus, (212,0), (340,12), black, -1)
#                 cv2.putText(menus,key1,(215,10),cv2.FONT_HERSHEY_PLAIN,0.7,classifc[key1],1 )
#                 labelfound=True
#                 break
#             posrc+=1

#         if  x> zoneverticalgauche[0][0] and y > zoneverticalgauche[0][1] and x<zoneverticalgauche[1][0] and y<zoneverticalgauche[1][1]:
#             print ( 'this is in menu')
#             labelfound=True

#         if  x> zoneverticaldroite[0][0] and y > zoneverticaldroite[0][1] and x<zoneverticaldroite[1][0] and y<zoneverticaldroite[1][1]:
#             print ('this is in menu')
#             labelfound=True

#         if  x> zonehorizontal[0][0] and y > zonehorizontal[0][1] and x<zonehorizontal[1][0] and y<zonehorizontal[1][1]:
#             print ('this is in menu')
#             labelfound=True

#         if x>posxdel and x<posxdel+10 and y>posydel and y< posydel+10:
#             print ('this is suppress')
#             # suppress()
#             labelfound=True

#         if x>posxquit and x<posxquit+10 and y>posyquit and y< posyquit+10:
#             print ('this is quit')
#             quitl=True
#             labelfound=True

#         if x>posxdellast and x<posxdellast+10 and y>posydellast and y< posydellast+10:
#             print ('this is delete last')
#             labelfound=True
#             # dellast()

#         if x>posxdelall and x<posxdelall+10 and y>posydelall and y< posydelall+10:
#             print ('this is delete all')
#             labelfound=True
#             # delall()

#         if x>posxcomp and x<posxcomp+10 and y>posycomp and y< posycomp+10:
#             print ('this is completed for all')
#             labelfound=True
#             # completed(imagename,dirpath_patient)

#         if x>posxreset and x<posxreset+10 and y>posyreset and y< posyreset+10:
#             print( 'this is reset')
#             labelfound=True
#             # reseted()
#         if x>posxvisua and x<posxvisua+10 and y>posyvisua and y< posyvisua+10:
#             print ('this is visua')
#             labelfound=True
#             # visua()
#         if x>posxeraseroi and x<posxeraseroi+10 and y>posyeraseroi and y< posyeraseroi+10:
#             print ('this is erase roi')
#             labelfound=True
#             # eraseroi(imagename,dirpath_patient)

#         if x>posxlastp and x<posxlastp+10 and y>posylastp and y< posylastp+10:
#             print ('this is last point')
#             labelfound=True
#             # closepolygon()



#         if not labelfound:
#             print ('add point',pattern)
#             if len(pattern)>0:
# #                print 'len pattern >0'
#                 print ('fxs,x0',fxs,x0new)
# #                global fxs,x0new,y0new
#                 xnew=int((x+x0new)/fxs)
#                 ynew=int((y+y0new)/fxs)
#                 print (x,y,xnew,ynew)
#                 numeropoly=tabroinumber[pattern][scannumber]
#                 print ('length last pattent',len(tabroi[pattern][scannumber][numeropoly]))
#                 tabroi[pattern][scannumber][numeropoly].append((xnew, ynew))
#                 print( numeropoly, tabroi[pattern][scannumber][numeropoly])
#                 cv2.rectangle(images[scannumber], (xnew,ynew),
#                               (xnew,ynew), classifc[pattern], 1)

#                 for l in range(0,len(tabroi[pattern][scannumber][numeropoly])-1):
#                     cv2.line(images[scannumber], (tabroi[pattern][scannumber][numeropoly][l][0],tabroi[pattern][scannumber][numeropoly][l][1]),
#                               (tabroi[pattern][scannumber][numeropoly][l+1][0],tabroi[pattern][scannumber][numeropoly][l+1][1]), classifc[pattern], 1)
#                     l+=1
# #                cv2.imshow('images',images[scannumber])
#             else:
#                 cv2.rectangle(menus, (212,0), (340,12), black, -1)
#                 cv2.putText(menus,'No pattern selected',(215,10),cv2.FONT_HERSHEY_PLAIN,0.7,white,1 )


def loop(ggdinit):
    global quitl
    quitl=False
   
    image=cv2.cvtColor(ggdinit,cv2.COLOR_BGR2RGB)

    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    # cv2.namedWindow("Slider2",cv2.WINDOW_NORMAL)

    # cv2.createTrackbar( 'Brightness','Slider2',0,100,nothing)
    # cv2.createTrackbar( 'Contrast','Slider2',50,100,nothing)
    # cv2.createTrackbar( 'Flip','Slider2',slnt/2,slnt-2,nothing)
    # cv2.createTrackbar( 'Zoom','Slider2',0,100,nothing)
    # cv2.createTrackbar( 'Panx','Slider2',50,100,nothing)
    # cv2.createTrackbar( 'Pany','Slider2',50,100,nothing)
    cv2.setMouseCallback("image", click_and_crop)

    while True:

        key = cv2.waitKey(100) & 0xFF
        if key != 255:
            print( key)

        if key == ord("c"):
                print ('completed')
                # completed(imagename,dirpath_patient)

        elif key == ord("d"):
                print ('delete entry')
                # suppress()

        elif key == ord("l"):
                print ('delete last polygon')
                # dellast()

        elif key == ord("a"):
                print( 'delete all')
                # delall()

        elif key == ord("r"):
                print ('reset')
                # reseted()

        elif key == ord("v"):
                print ('visualize')
                # visua()
        elif key == ord("e"):
                print ('erase')
                # eraseroi(imagename,dirpath_patient)
        elif key == ord("f"):
                print ('close polygone')
                # closepolygon()
        elif key == ord("q")  or quitl or cv2.waitKey(20) & 0xFF == 27 :
               print ('on quitte', quitl)
               cv2.destroyAllWindows()
               break
#         c = cv2.getTrackbarPos('Contrast','Slider2')
#         l = cv2.getTrackbarPos('Brightness','Slider2')
#         fl = cv2.getTrackbarPos('Flip','Slider2')
#         z = cv2.getTrackbarPos('Zoom','Slider2')
#         px = cv2.getTrackbarPos('Panx','Slider2')
#         py = cv2.getTrackbarPos('Pany','Slider2')
# #        print fl
#         scannumber=fl+1
#         imagename=list_image[scannumber]
#         imagenamecomplet=os.path.join(pdirk,imagename)
#         image = cv2.imread(imagenamecomplet,cv2.IMREAD_ANYDEPTH)
#         image=cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)

#         image=zoomfunction(image,z,px,py)
#         imagesview=zoomfunction(images[scannumber],z,px,py)

# #        initmenus(slnt,dirpath_patient,z)
#         populate(dirpath_patient,slnt)
#         imglumi=lumi(image,l)
#         image=contrasti(imglumi,c)
# #        print image.shape,images[scannumber].shape
#         imageview=cv2.add(image,imagesview)
#         imageview=cv2.add(imageview,menus)
#         for key1 in classif:
#                 tabroifinalview=zoomfunction(tabroifinal[key1][scannumber],z,px,py)
#                 imageview=cv2.add(imageview,tabroifinalview)
#         imageview=cv2.cvtColor(imageview,cv2.COLOR_BGR2RGB)
        cv2.imshow("image", image)
        
        
def overlay(arr,num,img,cx,cy):
    no = -1
    for i in range(9):
        for j in range(9):
            no += 1 
            #cv2.putText(img,str(no), (int(cx[i][j]),int(cy[i][j])),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            if num[no] == 0:
                
                cv2.putText(img,str(int(arr[j][i])), (int(cx[i][j]-4),int(cy[i][j])+8),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
                
    cv2.imshow("Sudoku",img)
    cv2.waitKey(0)
  
def check_col(arr,num,col):
    if  all([num != arr[i][col] for i in range(9)]):
        return True
    return False
    

def check_row(arr,num,row):
    if  all([num != arr[row][i] for i in range(9)]):
        return True
    return False


def check_cell(arr,num,row,col):
    sectopx = 3 * (row//3)
    sectopy = 3 * (col//3)
          
    for i in range(sectopx, sectopx+3):
        for j in range(sectopy, sectopy+3):
            if arr[i][j] == num:
                return True
    return False


def empty_loc(arr,l):
    for i in range(9):
        for j in range(9):
            if arr[i][j] == 0:
                l[0]=i
                l[1]=j
                return True              
    return False

#### Solving sudoku by back tracking############
def solvesudoku(arr):
    l=[0,0]

    if not empty_loc(arr,l):
        return True
    
    row = l[0]
    col = l[1]
                
    for num in range(1,10):
        if check_row(arr,num,row) and check_col(arr,num,col) and not check_cell(arr,num,row,col):
            arr[row][col] = int(num) 
            
            if(solvesudoku(arr)):
                return True
 
            # failure, unmake & try again
            arr[row][col] = 0
                    
    return False


def aff(tabref,tabaf,col):
    grid = np.zeros(image.shape, dtype=np.uint8)
    r=col[0]
    g=col[1]
    b=col[2]
    for y in range(0,9):
       for x in range(0,9):

            x0=int(cellLocs[x][y][0])
            x1=int(cellLocs[x][y][2])
            y0=int(cellLocs[x][y][1])
            y1=int(cellLocs[x][y][3])
            # print(x0,x1,y0,y1)
            
            px =int((x0+x1)/2)
            py=int((y0+y1)/2)
            if  tabaf[x,y] != 0 :
                if tabref[x,y] ==0 :
                    # print('diff',y,x,tabaf[x,y])
                    cv2.putText(grid, str(tabaf[x,y]),(px,py), font, 0.6, (r, g, b), 1, cv2.LINE_AA)
                else:
                    # print('equiv diff',y,x,tabaf[x,y],tabref[x,y])
                    cv2.putText(grid, str(tabaf[x,y]),(px,py), font, 0.5, (b, g, r), 1, cv2.LINE_AA)
            grid=cv2.rectangle(grid, (x1,y1), (x0,y0), (255,255,255), 2)
    return grid

def affinit(tabref,col):
    grid = np.zeros(image.shape, dtype=np.uint8)
    r=col[0]
    g=col[1]
    b=col[2]
    for y in range(0,9):
       for x in range(0,9):
            x0=int(cellLocs[x][y][0])
            x1=int(cellLocs[x][y][2])
            y0=int(cellLocs[x][y][1])
            y1=int(cellLocs[x][y][3])
            # print(x0,x1,y0,y1)
            
            px =int((x0+x1)/2)
            py=int((y0+y1)/2)
            
            if  tabref[x,y] != 0 :
                    cv2.putText(grid, str(tabref[x,y]),(px,py), font, 0.5, (b, g, r), 1, cv2.LINE_AA)
            grid=cv2.rectangle(grid, (x1,y1), (x0,y0), (255,255,255), 2)
    return grid
    
    
def lookForFig():
    # loop over the grid locations
    for y in range(0, 9):
    	# initialize the current list of cell locations
        row = []
        for x in range(0, 9):
            # print(x,y)
    		# compute the starting and ending (x, y)-coordinates of the
    		# current cell
            startX = x * stepX
            startY = y * stepY
            endX = (x + 1) * stepX
            endY = (y + 1) * stepY
    		# add the (x, y)-coordinates to our cell locations list
            row.append((startX, startY, endX, endY))
            # crop the cell from the warped transform image and then
    		# extract the digit from the cell
            #cell = warped[startY:endY, startX:endX]
            cell =  puzzleImage[startY:endY, startX:endX]
            cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    
            cellc=cell.copy()
            # print(np.mean(cellc))
            (h, w) = cellc.shape
            # np.putmask(cellc,cellc > 200,255)
            np.putmask(cellc,cellc < np.mean(cellc),0)
    
            percentFilled = np.count_nonzero(cellc) / float(w * h)
            # percentFilled = np.count_nonzero(cellc)
    
            # if (x==6 or x==8) and (y ==1):
            #             # print('boxes',boxes)
            #             print(percentFilled)
            #             # print(percentFilled)
            #             cv2.imshow("cell", cell)
            #             cv2.imshow("cellc", cellc)
            #             # cv2.imshow("Digit", roi)
            #             cv2.waitKey(0)
            #             cv2.destroyAllWindows()
    
    
            if percentFilled <0.5:
                cell=255-cell
    
            # print(percentFilled)
    
            # # print(percentFilled)
            # cv2.imshow("cell", cell)
            # # cv2.imshow("roi1", roi1)
            # # cv2.imshow("Digit", roi)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            digit = extract_digit(cell, debug=False)
    		# verify that the digit is not empty
            if digit is not None:
            # if True:
    			# resize the cell to 28x28 pixels and then prepare the
    			# cell for classification
                # roi = cv2.resize(digit, (100,100 ))
    
                # roi=255-roi
                roi=255-digit
                # np.putmask(roi,roi > 128,255)
                # np.putmask(roi,roi < 129,0)
                # roi1=roi.copy()
                # print(roi.shape)
                # print(np.unique(roi))
                # kernel = np.ones((3,3),np.uint8)
                # roi = cv2.dilate(roi,kernel,iterations = 1)
                # kernel = np.ones((4,4),np.uint8)
                # roi = cv2.erode(roi,kernel,iterations = 2)
                boxes = pytesseract.image_to_boxes(roi,lang="eng",config=' --psm 10 --oem 3 -c tessedit_char_whitelist=123456789')
                # boxes1 = pytesseract.image_to_boxes(roi1,config='--psm 10 --oem 3 -c tessedit_char_whitelist=123456789')
                # roi = roi.astype("float") / 255.0
                boxes=boxes.split(' ')
    
                # if (x==6 or x==8) and (y ==1):
                # if True:
    
                #             print('boxes',boxes)
                            # print(percentFilled)
                            # cv2.imshow("roi", roi)
                            # cv2.imshow("roi1s", 255-roi1)
                            # cv2.imshow("Digit", roi)
                            # cv2.waitKey(0)
                            # cv2.destroyAllWindows()
                            # cv2.imwrite(str(y)+'_'+str(x)+'.jpg',roi)
                    # if restext !=ref[y,x]:
                # boxes1=boxes1.split(' ')
    
            # # print(boxes,boxes[0])
    
                try:
                    restext=int(boxes[0])
                except:
                    # print(boxes)
                    restext=0
                 
                # if restext !=ref[y,x]:
                #     print(restext,boxes[0])
                #     cv2.imshow("roi", roi)
                #     cv2.waitKey(0)
                #     cv2.destroyAllWindows()
                  
                if restext in [1,2,3,4,5,6,7,8,9]:
                    # print(percentFilled)
                    # print(restext)
                    if False:
    
                            print('boxes',boxes)
                            print(x,y)
                            (h, w) = roi.shape
    
                            percentFilled = cv2.countNonZero(roi) / float(w * h)
                            print(percentFilled)
                            cv2.imshow("roi", roi)
                            # cv2.imshow("roi1", roi1)
                            # cv2.imshow("Digit", roi)
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()
                    # if restext !=ref[y,x]:
    
    
                            cv2.imshow("roi", roi)
                            # cv2.imshow("roi1", roi1)
                            # cv2.imshow("Digit", roi)
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()
                # roi = img_to_array(roi)
                # roi = np.expand_dims(roi, axis=0)
    			# classify the digit and update the Sudoku board with the
    			# prediction
                # pred = model.predict(roi).argmax(axis=1)[0]
                board[y, x] = restext
               	# add the row to our cell locations
        cellLocs.append(row)
    return board,cellLocs

print("[INFO] loading input image...")
image = cv2.imread(imageToLoad)

# image =rotate_image(image,90)

aspectratio=image.shape[0]/image.shape[1]
newheight=int(finalwidth*aspectratio)

image=cv2.resize(image, (finalwidth, newheight))
(puzzleImage, warped) = find_puzzle(image, debug=False)

# initialize our 9x9 Sudoku board
board = np.zeros((9, 9), dtype="int")
# a Sudoku puzzle is a 9x9 grid (81 individual cells), so we can
# infer the location of each cell by dividing the warped image
# into a 9x9 grid
stepX = warped.shape[1] // 9
stepY = warped.shape[0] // 9
# initialize a list to store the (x, y)-coordinates of each cell
# location
cellLocs = []
ref = np.zeros((9, 9), dtype="int")
# ref[1,8]=6
# ref[1,7]=2
# ref[2,5]=2
# ref[2,6]=3
# ref[2,7]=9
# ref[3,1]=5
# ref[3,7]=3
# ref[4,4]=3
# ref[4,6]=9
# ref[4,7]=8
# ref[5,2]=1
# ref[5,5]=8
# ref[5,7]=7
# ref[5,8]=2
# ref[6,2]=8
# ref[6,5]=7
# ref[6,7]=6
# ref[7,1]=6
# ref[7,2]=5
# ref[7,5]=1
# ref[7,6]=7
# ref[7,8]=3
# ref[8,1]=1
# ref[8,2]=7
# ref[8,4]=2
# ref[8,7]=5



ref[1,7]=4
ref[1,8]=3
ref[2,4]=3
ref[2,5]=1

ref[3,1]=4
ref[3,5]=7
ref[4,2]=3
ref[4,6]=2
ref[4,8]=8
ref[5,2]=8
ref[5,8]=5
ref[6,1]=9
ref[6,2]=4
ref[6,4]=8
ref[6,7]=5
ref[7,1]=3
ref[7,4]=5
ref[7,5]=9
ref[7,6]=7
ref[8,1]=8
ref[8,2]=5
ref[8,3]=7
ref[8,6]=1
ref[8,7]=2


board,cellLocs=lookForFig()
print(board)


tabresrinit=board.copy()
ggdinit=affinit(tabresrinit,(255,255,0))
loop(ggdinit)

if solve:
    if(solvesudoku(board)):
        ggd=aff(tabresrinit,board,(255,255,0))
    
    else:
        ggd=ggdinit
        print("There is no solution")


ggd=aff(tabresrinit,board,(255,255,0))

cv2.imshow('ggdinit', ggdinit)
cv2.imshow('gg_done', ggd)
cv2.imshow('image', image)
cv2.waitKey()
cv2.destroyAllWindows()  