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
font = cv2.FONT_HERSHEY_SIMPLEX

debug=True

finalwidth=600

# load the input image from disk and convert it to grayscale
print("[INFO] loading input image...")
image = cv2.imread('sudoku1.jpg')
# image = cv2.imread('s1.jpg')
# image = cv2.imread('s2.png')

aspectratio=image.shape[0]/image.shape[1]
newheight=int(finalwidth*aspectratio)

image=cv2.resize(image, (finalwidth, newheight))

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
ref[1,8]=6
ref[1,7]=2
ref[2,5]=2
ref[2,6]=3
ref[2,7]=9
ref[3,1]=5
ref[3,7]=3
ref[4,4]=3
ref[4,6]=9
ref[4,7]=8
ref[5,2]=1
ref[5,5]=8
ref[5,7]=7
ref[5,8]=2
ref[6,2]=8
ref[6,5]=7
ref[6,7]=6
ref[7,1]=6
ref[7,2]=5
ref[7,5]=1
ref[7,6]=7
ref[7,8]=3
ref[8,1]=1
ref[8,2]=7
ref[8,4]=2
ref[8,7]=5

# loop over the grid locations
for y in range(0, 9):
	# initialize the current list of cell locations
    row = []
    for x in range(0, 9):
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
        cell = warped[startY:endY, startX:endX]
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
            roi = cv2.resize(digit, (64, 64))

            roi=255-roi
            # np.putmask(roi,roi > 128,255)
            # np.putmask(roi,roi < 129,0)
            roi1=roi.copy()
            # print(roi.shape)
            # print(np.unique(roi))
            kernel = np.ones((3,3),np.uint8)
            roi = cv2.dilate(roi,kernel,iterations = 1)
            kernel = np.ones((4,4),np.uint8)
            roi = cv2.erode(roi,kernel,iterations = 1)
            boxes = pytesseract.image_to_boxes(roi,config='--psm 10 --oem 3 -c tessedit_char_whitelist=123456789')
            # boxes1 = pytesseract.image_to_boxes(roi1,config='--psm 10 --oem 3 -c tessedit_char_whitelist=123456789')
            # roi = roi.astype("float") / 255.0
            boxes=boxes.split(' ')

            # if (x==6 or x==8) and (y ==1):
            #             print('boxes',boxes)
            #             # print(percentFilled)
            #             cv2.imshow("roi", roi)
            #             cv2.imshow("roi1", roi1)
            #             # cv2.imshow("Digit", roi)
            #             cv2.waitKey(0)
            #             cv2.destroyAllWindows()
                # if restext !=ref[y,x]:
            # boxes1=boxes1.split(' ')

        # # print(boxes,boxes[0])

            try:
                restext=int(boxes[0])
            except:
                # print(boxes)
                restext=0

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
                        cv2.imshow("roi1", roi1)
                        # cv2.imshow("Digit", roi)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                # if restext !=ref[y,x]:


                        cv2.imshow("roi", roi)
                        cv2.imshow("roi1", roi1)
                        # cv2.imshow("Digit", roi)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
            # roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
			# classify the digit and update the Sudoku board with the
			# prediction
            # pred = model.predict(roi).argmax(axis=1)[0]
            board[y, x] = restext
	# add the row to our cell locations
    cellLocs.append(row)

# print(board)

   
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

# tabresr[0,0]=2
# tabresr[2,0]=8

tabresrinit=board.copy()



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
            if  tabaf[y,x] != 0 :
                if tabref[x,y] ==0 :
                    # print('diff',y,x,tabaf[x,y])
                    cv2.putText(grid, str(tabaf[x,y]),(px,py), font, 0.6, (r, g, b), 1, cv2.LINE_AA)
                else:
                    # print('equiv diff',y,x,tabaf[x,y],tabref[x,y])
                    cv2.putText(grid, str(tabaf[x,y]),(px,py), font, 0.5, (b, g, r), 1, cv2.LINE_AA)
            grid=cv2.rectangle(grid, (x1,y1), (x0,y0), (255,255,255), 2)
    return grid



# hist = cv2.equalizeHist(gray)

# # apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
# print("[INFO] applying CLAHE...")
# clahe = cv2.createCLAHE(clipLimit=2.0,
#  	tileGridSize=(8, 8))
# equalized = clahe.apply(gray)

# show the original grayscale image and CLAHE output image

# cv2.imshow("Input", gray)
# cv2.imshow("thresh", thresh)
# # cv2.imshow("hist", hist)
# cv2.waitKey(0)
# cv2.imshow("Puzzle Transform", puzzle)
# cv2.imshow("warped", warped)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
if(solvesudoku(board)):

    ggd=aff(tabresrinit,board,(255,255,0))

else:
    print("There is no solution")

# if(solvesudoku(tabresr)):
#     arr = board(grid)
#     overlay(arr,num,warped1,cx,cy)

# else:
#     print("There is no solution")    
# tabresrsolved= np.zeros((9,9), dtype=np.uint8)
    
# print(tabresr)
ggd=aff(tabresrinit,board,(255,255,0))
# tabresrsolved=solvesudoku(tabresr)
# print(tabresr)
# print(tabzonev)
    
# cv2.imshow('tabresr', tabresr)

# cv2.imshow('gg_init', gg)
cv2.imshow('gg_done', ggd)
cv2.imshow('image', image)
cv2.waitKey()
cv2.destroyAllWindows()  