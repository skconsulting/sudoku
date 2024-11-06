# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 18:20:38 2021
environment: my-env

@author: sylvain
"""

# USAGE
# python adaptive_equalization.py --image images/boston.png
# tablist list possible
  # keys up,down,left,right: move position one case
  # + add number
  # - substract number from possible
  # ctrl z : undo
  # Suppr: suppress entry in modifu
  # d visualise double
  # H Help
  # h reset Help
  # l   reset list all possibles
  # L toggle view list possible 
  # m modify source
  # n new blank
  # q or Q: quit   
  # r reset all
  # s save to save.p
  # S solve
  # t visualise triple
  # v + "number" visualise all possible "number"
  # x write for Simple Sudoku
  # w webcam



from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import numpy as np
import imutils ##scikit-image
import cv2
import pytesseract
import os
# from appJar import gui
import pickle
import time
from sudoku import Sudoku ##py-sudoku
import random
import itertools

cwd= os.path.dirname(__file__)


font = cv2.FONT_HERSHEY_SIMPLEX

debug=False
solve=False

readJPG=False # read images or not if True  read image else read saved sata
Radomseed=False

finalwidth=600
imageToLoad='sudoku1.jpg'
#imageToLoad='expert.jpg'
imageToLoad='cam.jpg'
# imageToLoad='sudoku2_td.jpg'


CaseSelected = False
pxSelected = -1
pySelectedt = -1


# imageToLoad='s1.jpg'
# imageToLoad='s2.jpg'
#imageToLoad='sexpert.jpg'
# imageToLoad='s2.png'

def rotate_image(image, angle):
  image_center = tuple(np.array(shapeRef[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, shapeRef[1::-1], flags=cv2.INTER_LINEAR)
  return result

def find_puzzle(image, debug=False):
    print("[INFO] get puzzle")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # ORIG blurred = cv2.GaussianBlur(gray, (7, 7), 3)
    blurred = cv2.GaussianBlur(gray, (7, 7), 1)
    # apply adaptive thresholding and then invert the threshold map
    thresh = cv2.adaptiveThreshold(blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)
    if debug:
        cv2.imshow("Puzzle Thresh", thresh)
        cv2.waitKey(0)
        #cv2.destroyAllWindows()
        cv2.destroyWindow("Puzzle Thresh")
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
        #cv2.destroyAllWindows()
        cv2.destroyWindow("Puzzle Outline")
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
            cv2.destroyWindow("Puzzle Transform")
            cv2.destroyWindow("warped")
    print("[INFO] puzzle ready")
    return (puzzle, warped)


def extract_digit(cell, debug=False):
    #print("start extract digit")
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
        cv2.destroyWindow("Cell Thresh")
        cv2.destroyWindow("Digit")
    # return the digit to the calling function
    #print("end extract digit")
    return digit

def moinslundo(px,py,num):
    global tablist,tabentert
    listp=tablist[px,py]
    listp.append(num)
    tablist[px,py]=listp
    #print('remove',num)
    # imagel=lfp(imagel)



def moinsl(px,py,num):
    global tablist,tabentert
    if tabentert[px,py]==num:
        tabentert[px,py]=0
    else:
        listp=tablist[px,py]
        if num in listp:
            listp.remove(num)
            tablist[px,py]=listp
            #print('remove',num)
            # imagel=lfp(imagel)
        else:
            print('value ' ,value,' not in list')
                
def plusl(px,py,num):
    global tabentert
    if check_row(tabentert,num,px) and check_col(tabentert,num,py) and not check_cell(tabentert,num,px,py):
        tabentert[px,py]=num
        #print('add',num)
    else:
        # print('value ' ,num,' impossible for row',check_row(tabentert,num,px) )
        # print('value ' ,num,' impossible for col',check_col(tabentert,num,py) )
        # print('value ' ,num ,' impossible for cell',not(check_cell(tabentert,num,px,py)) )
        print('value ' ,num,' impossible')
    
def pluslmos(px,py,num,board):
        if num != 0:
            if check_row(board,num,px) and check_col(board,num,py) and not check_cell(board,num,px,py):
                board[px,py]=num
                #print('add',num)
            else:
                # print('value ' ,num,' impossible for row',check_row(tabentert,num,px) )
                # print('value ' ,num,' impossible for col',check_col(tabentert,num,py) )
                # print('value ' ,num ,' impossible for cell',not(check_cell(tabentert,num,px,py)) )
                print('value ' ,num,' impossible for board')
        else:
            board[px,py]=0
        return board
    
def click_and_crop(event, x, y, flags, param):
    global quitl,lookforList,tabentert,plus,value,visu,gridVisu,tablist,ArraySelected,CaseSelected,moins,px,py,dbclick,command
    # command=''

    if event == cv2.EVENT_LBUTTONDOWN:
        # cv2.rectangle(menus, (150,12), (370,32), black, -1)
        # posrc=0
        # print ("x:",x,"y:",y)
        if  x> 0 and y > 0 and x<30 and y<30:
               print ( 'this is 1')
               cchiffre(1)
        elif  x> 35 and y > 0 and x<65 and y<30:
                print ( 'this is 2')
                cchiffre(2)
        elif  x> 70 and y > 0 and x<100 and y<30:
                print ( 'this is 3')
                cchiffre(3)
        elif  x> 105 and y > 0 and x<135 and y<30:
                 print ( 'this is 4')
                 cchiffre(4)
        elif  x> 140 and y > 0 and x<170 and y<30:
                  print ( 'this is 5')
                  cchiffre(5)
        elif  x> 175 and y > 0 and x<205 and y<30:
          print ( 'this is 6')
          cchiffre(6)
        elif  x> 210 and y > 0 and x<240 and y<30:
          print ( 'this is 7')
          cchiffre(7)
        elif  x> 245 and y > 0 and x<275 and y<30:
            print ( 'this is 8')
            cchiffre(8)
        elif  x> 280 and y > 0 and x<310 and y<30:
                print ( 'this is 9')
                cchiffre(9)
        elif  x> 360 and y > 0 and x<390 and y<30:
               # print ( 'this is +')
               # command='+'
               cplus()
        elif  x> 400 and y > 0 and x<430 and y<30:
                # print ( 'this is -')
                # command='+'
                cmoins()
        elif  x> 440 and y > 0 and x<470 and y<30:
                 # print ( 'this is -')
                 # command='+'
                 cvalue()
        elif  x> 480 and y > 0 and x<510 and y<30:
                  # print ( 'this is -')
                  # command='+'
                  cdouble()
        elif  x> 570 and y > 0 and x<600 and y<30:
            quitl=True
            print ('on quitte', quitl)
            cv2.destroyAllWindows()
        elif x>0 and x<30 and y >40 and y <70:
            print('start webcam')
            cwebcam()

        elif x>35 and x<65 and y >40 and y <70:
             print('save webcam')
             sawebcam()

        elif x>70 and x<100 and y >40 and y <70:
            print('ok webcam')
            okwebcam()

        elif x>105 and x<135 and y >40 and y <70:
           print('nok webcam')
           nokwebcam()
        elif x>140 and x<170 and y >40 and y <70:
            print('quit webcam')
            qwebcam()
         
        else:
       #             labelfound=True
            px,py,ArraySelected,CaseSelected =lookForPxPy(x,y)
         
            redraw()
    if event == cv2.EVENT_LBUTTONDBLCLK:

        # cv2.rectangle(menus, (150,12), (370,32), black, -1)
        # posrc=0
        #print ("x:",x,"y:",y)
        #print('this is double click')
        dbclick=True
        doubleclick()
 

def lfp(tabentert):
    # global tabentert
    col=(220,128,64)
    grid = np.zeros(shapeRef, dtype=np.uint8)
    tablist={}
    for x in range(0,9):
       for y in range(0,9):
           tablist[(x,y)]=()
    # print( 'list all')
    for x in range(0,9):
       for y in range(0,9):
           if tabentert[x,y]==0:
               listp=lookForPossible(x,y,tabentert)
               tablist[(x,y)]=listp
               grid+=affichList(x,y,listp,col)
    return grid,tablist

def lfplist(tabentert):
    col=(220,128,64)
    grid = np.zeros(shapeRef, dtype=np.uint8)
    # print( 'list all')
   
    for x in range(0,9):
       for y in range(0,9):
           listf=[]
            
           if (x,y) in tablist:
               listp=tablist[(x,y)]
               listp1=lookForPossible(x,y,tabentert)
               for u in listp: 
                   if u in listp1:
                       if tabentert[x,y]==0:
                           listf.append(u)
               tablist[(x,y)]=listf
                   

               grid+=affichList(x,y,listf,col)
    return grid
        # delall()
        # delall()

def rempliMenu()    :
    global gridMenu
    for i in range(10):
     
         cv2.rectangle(gridMenu, ((i*30)+5*i,0), ((i*30)+5*i+30,30), (100,100,100), -1)
         cv2.putText(gridMenu,str((i+1)%10) ,((i*30)+5*i+5,15), cv2.FONT_HERSHEY_PLAIN,0.7,(240,200,180),1)

    cv2.rectangle(gridMenu, (360,0), (390,30), (100,100,100), -1)
    cv2.putText(gridMenu,'+' ,(370,20), cv2.FONT_HERSHEY_PLAIN,1.0,(255,255,255),1)
    cv2.rectangle(gridMenu, (400,0), (430,30), (100,100,100), -1)
    cv2.putText(gridMenu,'-' ,(410,20), cv2.FONT_HERSHEY_PLAIN,1.0,(255,255,255),1)
    cv2.rectangle(gridMenu, (440,0), (470,30), (100,100,100), -1)
    cv2.putText(gridMenu,'v' ,(450,20), cv2.FONT_HERSHEY_PLAIN,1.0,(255,255,255),1)
    cv2.rectangle(gridMenu, (480,0), (510,30), (100,100,100), -1)
    cv2.putText(gridMenu,'d' ,(490,20), cv2.FONT_HERSHEY_PLAIN,1.0,(255,255,255),1)
    cv2.rectangle(gridMenu, (570,0), (600,30), (100,100,100), -1)
    cv2.putText(gridMenu,'q' ,(580,20), cv2.FONT_HERSHEY_PLAIN,1.0,(255,255,255),1)
    
    cv2.rectangle(gridMenu, (0,40), (30,70), (100,100,100), -1)
    cv2.putText(gridMenu,'wc' ,(0,60), cv2.FONT_HERSHEY_PLAIN,1.0,(255,255,255),1)
    cv2.rectangle(gridMenu, (35,40), (65,70), (100,100,100), -1)
    cv2.putText(gridMenu,'swe' ,(35,60), cv2.FONT_HERSHEY_PLAIN,1.0,(255,255,255),1)
    cv2.rectangle(gridMenu, (70,40), (100,70), (100,100,100), -1)
    cv2.putText(gridMenu,'ok' ,(70,60), cv2.FONT_HERSHEY_PLAIN,1.0,(255,255,255),1)
    cv2.rectangle(gridMenu, (105,40), (135,70), (100,100,100), -1)
    cv2.putText(gridMenu,'nok' ,(105,60), cv2.FONT_HERSHEY_PLAIN,1.0,(255,255,255),1)
    cv2.rectangle(gridMenu, (140,40), (170,70), (100,100,100), -1)
    cv2.putText(gridMenu,'qw' ,(140,60), cv2.FONT_HERSHEY_PLAIN,1.0,(255,255,255),1)
    # cv2.rectangle(gridMenu, ((i*30)+5*i,50), ((i*30)+5*i+50,80), (100+i*10,256-(i*20),100), -1)
    # cv2.putText(gridMenu,str(i+1) ,((i*30)+5*i+5,15), cv2.FONT_HERSHEY_PLAIN,0.7,(240,200,180),1)
    
def redraw():
        global showPossible,gridpair,tabentertimg,showPossible,image,gridVisu,ArraySelected,gridMenu
        global tabentert,tabresrinit,imagel,valueVisu,visuActive,toggleDou
        global lastpx,lastpy,tabhelp,errormoins,gridColor
       # print('I redraw')
        griderror=np.zeros(shapeRef,np.uint8)
        resultSolved,_=solvesudokuNew(tabentert)
        if not (resultSolved) or errormoins:
            print('error ',"resultSolved",resultSolved," errormoins ",errormoins)
            if lastpx>-1 and lastpy>-1:
                griderror+=visui(lastpx,lastpy,(0,0,255))    
        # else:   
        #         print('NO error ')
        
        tabentertimg=affinit(tabentert,(125,201,10))

        imagel=lfplist(tabentert)
        # print('imagel',imagel.shape)
        # print('image',image.shape)
        if showPossible:
            imageview=cv2.add(image,imagel) ## image = source, imagel is possible
        else:
            imageview=image
        visiuNum(valueVisu,visuActive)
        #togdou(toggleDou)
        imageview=cv2.add(imageview,gridVisu) ## highlight only some number
        imageview=cv2.add(imageview,griderror) ## highlight error
        imageview=cv2.add(imageview,tabentertimg) ## with entered num
        imageview=cv2.add(imageview,ArraySelected)## cell selected
        imageview=cv2.add(imageview,gridpair)## pairs
        imageview=cv2.add(imageview,tabhelp) # help
        imageview=cv2.add(imageview,gridColor)## colorisation

#         imageview=cv2.cvtColor(imageview,cv2.COLOR_BGR2RGB)

        finish=True
        for x in range (0,9):
            for y in range(0,9):
                if tabentert[x][y]==0 and tabresrinit[x][y]==0:
                    finish=False
                    break
        if finish:
            print('FINISH')
            cv2.rectangle(gridMenu, (400,27), (550,58), (20,30,10), -1)
            cv2.putText(gridMenu,'FINISH' ,(450,50), cv2.FONT_HERSHEY_PLAIN,2,(240,200,180),1)
        imageview=np.concatenate((gridMenu,imageview),axis=0)
        cv2.imshow("image", imageview)
        
def doubleclick():
    global dbclick,CaseSelected,px,py,value,tabhelp
    if dbclick:
        if  CaseSelected:
            listp=tablist[px,py]
            if len(listp) ==1:
                value= tablist[px,py][0]
                print('I add', value)
                tabhelp=np.zeros(shapeRef,np.uint8)
              #  plus=False
                plusl(px,py,value)
                value = -1
                redraw()
    dbclick=False
        

def drawselected(px,py):
    ArraySelected =np.zeros( np.shape(image), dtype=np.uint8)
    start_point = (cellLocs[px][py][0], cellLocs[px][py][1]) 
    end_point = (cellLocs[px][py][2], cellLocs[px][py][3]) 
    # print("start_point",start_point)
    # print("end_point",end_point)
    color=(255,255,0)
    thickness=3
    ArraySelected = cv2.rectangle(ArraySelected, start_point, end_point, color, thickness) 
    return(ArraySelected)

def visiuNum(valueVisu,toggleVisu):
    global gridVisu,tabentert,tablist
   # print('visu',valueVisu,toggleVisu)

    gridVisu=np.zeros(shapeRef,np.uint8)
    # print(tabentert)
    # visu=False
    if valueVisu>0 and toggleVisu:
        for x in range (0,9):
            for y in range(0,9):
                if tabentert[x][y]==valueVisu:
                    gridVisu+=visui(x,y,(150,150,150))
                    # print('got',x,y)
                if (x,y) in tablist:
                    listp=tablist[(x,y)]
                    if valueVisu in listp:
                        gridVisu+=visui(x,y,(80,120,80))                        
    #redraw()
def  togdou(toggleDou,n):
    global tablist,gridpair
    gridpair=np.zeros(shapeRef,np.uint8)
    if toggleDou:
        for x in range (0,9):
            for y in range(0,9):
                    # print('got',x,y)
                if (x,y) in tablist:
                    listp=tablist[(x,y)]
                    if len(listp)==n:
                        gridpair+=visui(x,y,(80,80,80))          
def cplus():
    global mos,value,plus,visu,moins,dou
    print ('plus')
    mos=False
    value=-1
    plus=True
    visu=False
    moins=False
    dou=False
    
    cv2.rectangle(gridMenu, (400,30), (550,60), (20,30,10), -1)
    cv2.putText(gridMenu,'+' ,(450,55), cv2.FONT_HERSHEY_PLAIN,2,(240,200,180),1)
    redraw()

def cmoins():
  global mos,value,plus,visu,moins,dou,gridMenu
  print ('moins')
  mos=False
  value=-1
  moins=True
  plus=False
  visu=False
  dou=False
  cv2.rectangle(gridMenu, (400,30), (590,60), (20,30,10), -1)
  #◄cv2.rectangle(gridMenu, (400,27), (550,58), (20,30,10), -1)
  cv2.putText(gridMenu,'-' ,(450,55), cv2.FONT_HERSHEY_PLAIN,2,(240,200,180),1)
  redraw()
  
def cvalue():
       global mos,value,plus,visu,moins,dou,visuActive,gridpair,gridMenu
       value=-1
       visu=True
       # waitforvalue=True
       visuActive = not(visuActive)
       #print("visuActive",visuActive)
       # if visuActive:
       #      cv2.rectangle(gridMenu, (390,65), (550,95), (20,20,10), -1)
       #      cv2.putText(gridMenu,'visu : '+ ' ' ,(410,90), cv2.FONT_HERSHEY_PLAIN,2,(240,200,180),1)
       # else:
       #      cv2.putText(gridMenu,'           ' ,(410,90), cv2.FONT_HERSHEY_PLAIN,2,(240,200,180),1)

       # print ('visualize only some values',visuActive)
       moins=False
       plus=False
       dou=False
       gridpair=np.zeros(shapeRef,np.uint8)
       visugra(valueVisu,visuActive)
       redraw()
       # tabhelp=np.zeros(shapeRef,np.uint8)
def sawebcam():
    print('s webcam')
    global frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # print("Converted RGB image to grayscale...")
    # print("Resizing image to 28x28 scale...")
    img_ = cv2.resize(gray,(640,480))
    # print("Resized...")
    cv2.imwrite(filename=cwd+'/cam.jpg', img=img_)
    print("Image saved!")

    #image = cv2.imread('cam.jpg')
    cv2.imshow("Captured", img_)
    print('type "y" if correct')
    
def nokwebcam():
    cv2.destroyWindow("Captured")
    print('captured not good')
    
def qwebcam():
    global webcamLaunched,finl,webcam
    webcam.release()
    webcamLaunched=False
    finl=False
    print("Camera off.")
    cv2.destroyWindow("Capturing")
    #cv2.destroyAllWindows()

    
def okwebcam():
    global webcam,image,board,cellLocs,tabresrinit,tabentert,imagel,tablist,resultsolved,solvedFTrue,solved,ggdinit
    global tabentertimg,webcamLaunched,finl
    print('ok webcam')
    if webcamLaunched:
        webcam.release()
        board,cellLocs=loadimage('cam.jpg')
        cv2.destroyWindow("Capturing")
        cv2.destroyWindow("Captured")
    
    
        tabresrinit=board.copy()
        ggdinit=affinit(tabresrinit,(255,255,0))
        image=cv2.cvtColor(ggdinit,cv2.COLOR_BGR2RGB)
        tabentert=board.copy()
        tabentertimg=affinit(tabentert,(125,201,10))
    
        imagel,tablist=lfp(tabentert)
        # # cv2.imshow("puzzleImage", puzzleImage)
        # # cv2.imshow("warped", warped)
        # cv2.imshow("image", image)
        # cv2.setMouseCallback("image", click_and_crop)
        #cv2.waitKey()
        # cv2.destroyWindow("image")
        resultsolved,solved=solvesudokuNew(board)
        solvedFTrue=np.zeros((9, 9), dtype="int")
        for i in range(9):
           for j in range(9):
               solvedFTrue[i,j]=solved[i,j]
        redraw()
        finl=False
    else:
        print('need webcam launched')


def cdouble():
    global dou,toggleDou,moins,visu,plus,visuActive,gridMenu,tri
    dou=True
    tri=False
    toggleDou = not(toggleDou)
    print ('visualize double',toggleDou)
    moins=False
    plus=False
    visu=False
    visuActive=False
    # print(tabentert)
    cv2.rectangle(gridMenu, (400,60), (550,90), (20,30,10), -1)
    togdou(toggleDou,2)
    if toggleDou:
        cv2.putText(gridMenu,'double' ,(410,85), cv2.FONT_HERSHEY_PLAIN,2,(240,200,180),1)          
    dou=False
    redraw()
    
    
def ctriple():
    global dou,tri,toggleTri,moins,visu,plus,visuActive,gridMenu
    dou=False
    tri=True
    toggleTri = not(toggleTri)
    print ('visualize triple',toggleTri)
    moins=False
    plus=False
    visu=False
    visuActive=False
    # print(tabentert)
    cv2.rectangle(gridMenu, (400,60), (550,90), (20,30,10), -1)
    togdou(toggleTri,3)
    if toggleTri:
        cv2.putText(gridMenu,'triple' ,(410,85), cv2.FONT_HERSHEY_PLAIN,2,(240,200,180),1)      
    redraw()

def cwebcam():
    global webcamV,image,tabentertimg,tabentert,imagel,tablist,board,cellLocs,tabresrinit,ggdinit,solved
    global webcamLaunched,webcam,finl,frame

    finl=True
    webcamLaunched=True
    # print('start webcam')
    webcamV=False
    webcam = cv2.VideoCapture(0)

#try:

    while finl:
            check, frame = webcam.read()
            # print(check) #prints true as long as the webcam is running
            # print(frame) #prints matrix values of each framecd 
            #cv2.destroyWindow("image")
            cv2.imshow("Capturing", frame)
            key_ = cv2.waitKey(1)
            if key_ == ord('s'): 
                sawebcam()
                key_ = cv2.waitKey(0)
                if key_ == ord('y'): 
                    okwebcam()
                    break
                else:
                    nokwebcam()
                  
                    #webcam = cv2.VideoCapture(0)



            elif key_ == ord('q'):
                qwebcam()
              

def cchiffre(n):
    global command,mos,plus,visu,moins,dou,valueVisu,numsaved,errormoins,tabhelp,tri
    global lastpx,lastpy,tablistSaved,value,CaseSelected,toggleDou,gridMenu,visuActive,toggleTri,lastAction
    # print('cchiffre',px,py,moins,plus)
    
    if visu:
        valueVisu=n
        value=-1
  
    if moins:
        if px >-1 and py>-1:
            print ('moins ', n)
            value=n
            numsaved= n
            plus=False
            # waitforvalue=False
            lastpx=px
            lastpy=py
            tablistSaved= tablist.copy()
            lastAction="moins"
            cv2.rectangle(gridMenu, (400,30), (590,60), (20,30,10), -1)
            if solvedFTrue[px,py]==n:
                errormoins=True
                cv2.putText(gridMenu,'ERROR -'+': '+str(n) ,(400,55), cv2.FONT_HERSHEY_PLAIN,2,(0,0,250),1)
            else:
                errormoins=False
                cv2.putText(gridMenu,'- '+': '+ str(n) ,(450,55), cv2.FONT_HERSHEY_PLAIN,2,(240,200,180),1)
            tabhelp=np.zeros(shapeRef,np.uint8)
    if mos:
        # waitforvalue=False
        lastpx=px
        lastpy=py
        tablistSaved= tablist.copy()
        lastAction="mos"
        print ('add to source',value)
        
    if plus and n >0 and CaseSelected:
        # print ('plus', n)
        lastpx=px
        lastpy=py
        tablistSaved= tablist.copy()
        lastAction="plus"
        cv2.rectangle(gridMenu, (400,30), (550,60), (20,30,10), -1)
        cv2.putText(gridMenu,'+ '+': '+str(n) ,(450,55), cv2.FONT_HERSHEY_PLAIN,2,(240,200,180),1)
        tabhelp=np.zeros(shapeRef,np.uint8)
        moins=False
        value=n
        print('I add', value)
       #  plus=False
        plusl(px,py,value)
        value = -1
        plus=False
        redraw()
    if moins and n >0 and CaseSelected:
       #  moins=False
         moinsl(px,py,value)
         value = -1
         redraw()
    if visu:
        # valueVisu=value
        toggleDou=False
        toggleTri=False
        dou=False
        tri=False
        value=-1
        visugra(valueVisu,visuActive)
        # print('visu',valueVisu,visuActive)
        # cv2.putText(gridMenu,'visu : '+ ' ' ,(410,90), cv2.FONT_HERSHEY_PLAIN,2,(240,200,180),1)

        # if visuActive and (valueVisu in range(1,10)):
        #     cv2.rectangle(gridMenu, (390,70), (550,90), (20,20,10), -1)
        #     cv2.putText(gridMenu,'visu : '+ str(valueVisu) ,(410,95), cv2.FONT_HERSHEY_PLAIN,2,(240,200,180),1)
        # else:
        #     cv2.putText(gridMenu,'              ' ,(410,90), cv2.FONT_HERSHEY_PLAIN,2,(240,200,180),1)
            #•cv2.putText(gridMenu,'visu : '+ '7' ,(410,80), cv2.FONT_HERSHEY_PLAIN,2,(240,200,180),1)

        redraw()
def visugra(valueVisu,visuActive):
  global gridMenu
  
  cv2.rectangle(gridMenu, (390,65), (550,95), (20,20,10), -1)
  # cv2.putText(gridMenu,'visu : '+ ' ' ,(410,90), cv2.FONT_HERSHEY_PLAIN,2,(240,200,180),1)
  if visuActive :
      cv2.putText(gridMenu,'visu : ',(410,90), cv2.FONT_HERSHEY_PLAIN,2,(240,200,180),1)
      if (valueVisu in range(1,10)):
      # cv2.putText(gridMenu,'visu : '+ ' ' ,(410,80), cv2.FONT_HERSHEY_PLAIN,2,(240,200,180),1)
          cv2.putText(gridMenu,'visu : '+ str(valueVisu) ,(410,90), cv2.FONT_HERSHEY_PLAIN,2,(240,200,180),1)
  else:
      cv2.putText(gridMenu,'              ' ,(410,90), cv2.FONT_HERSHEY_PLAIN,2,(240,200,180),1)
      #•cv2.putText(gridMenu,'visu : '+ '7' ,(410,80), cv2.FONT_HERSHEY_PLAIN,2,(240,200,180),1)
      
    
def loop(board):
    global quitl,lookForList,plus,imagel,tabentert,value,tablist,moins,visu,gridVisu,gridMenu,ArraySelected,CaseSelected,px,py,gridpair,tabentertimg,showPossible
    global image,tabresrinit,valueVisu,visuActive,toggleDou,lastpx,lastpy,tabhelp,errormoins,solvedFTrue,gridColor
    global command,mos,dou,webcamLaunched,toggleTri,tri
    global key,numsaved,tablistSaved,webcamV,tabentertimg,lastAction

    ggdinit=affinit(board,(255,255,0))
    quitl=False
    lookForList=False
    plus=False
    moins=False
    visu=False
    dou=False
    mos=False
    webcamV=False
    toggleDou=False
    toggleTri=False
    tri=False
    showPossible=True
    visuActive=False
    color=False
    webcamLaunched=False
    value=-1
    lastpx=-1
    lastpy=-1
    px=-1
    py=-1
    valueVisu=-1
    errormoins=False
    command=''
    # waitforvalue=False
    image=cv2.cvtColor(ggdinit,cv2.COLOR_BGR2RGB)
    # print('image',image.shape)
    #oldshape=image.shape
    #print(oldshape)
    
    aspectratio=image.shape[0]/image.shape[1]
    newheight=int(finalwidth*aspectratio)

    image=cv2.resize(image, (finalwidth, newheight))
    # print('image new',image.shape)
   
    gridMenu=np.zeros((100,finalwidth,3),np.uint8)
    rempliMenu()
    # image=np.concatenate((image,gridMenu),axis=0)
    #newshape=image.shape
    #print("newshape",newshape)
    gridVisu=np.zeros(shapeRef,np.uint8)
    gridpair=np.zeros(shapeRef,np.uint8)
    tabhelp=np.zeros(shapeRef,np.uint8)
    imagel=np.ones(shapeRef,np.uint8)
    gridColor=np.zeros(shapeRef,np.uint8)
    
    
    
    # imagehighlight=np.zeros(shapeRef,np.uint8)
    height, width = image.shape[:2]

    #cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    cv2.namedWindow('image',cv2.WINDOW_AUTOSIZE)
    # cv2.resizeWindow('image', width, height)
    # cv2.namedWindow("Slider2",cv2.WINDOW_NORMAL)

    # cv2.createTrackbar( 'Brightness','Slider2',0,100,nothing)
    # cv2.createTrackbar( 'Contrast','Slider2',50,100,nothing)
    # cv2.createTrackbar( 'Flip','Slider2',slnt/2,slnt-2,nothing)
    # cv2.createTrackbar( 'Zoom','Slider2',0,100,nothing)
    # cv2.createTrackbar( 'Panx','Slider2',50,100,nothing)
    # cv2.createTrackbar( 'Pany','Slider2',50,100,nothing)
    cv2.setMouseCallback("image", click_and_crop)
    tabentertimg=affinit(tabentert,(125,201,10))
    # imagel,tablist=lfp()
    redraw()
    # print(tablist)

    while True:
        # keys up,down,left,right: move position one case
        # + add number
        # - substract number from possible
        # ctrl z : undo
        # Suppr: suppress entry in modifu
        # d visualise double
        # H Help
        # h reset Help
        # l   reset list all possibles
        # L toggle view list possible 
        # m modify source
        # n new blank
        # q or Q: quit   
        # r reset all
        # s save to save.p
        # S solve
        # t visualise triple
        # v + "number" visualise all possible "number"
        # x write for Simple Sudoku
        # w webcam
       # key = cv2.waitKey(100) & 0xFF


       
        key = cv2.waitKeyEx(0)
        # print('command',command)
        if key != 255:
            print( "I have typed",key)
            #up 2490368
            # down 2621440
            # right 2555904
            # left 2424832
        if key == 2490368:
            # print('this is up')
            if CaseSelected:
                px = max(0,px-1)
                ArraySelected=drawselected(px,py)
                value = -1
                redraw()
        elif key == 2621440:
             # print('this is down')
             if CaseSelected:
                 px = min(8,px+1)
                 ArraySelected=drawselected(px,py)
                 value = -1
                 redraw()
        elif key == 2555904:
             # print('this is right')
             if CaseSelected:
                 py = min(8,py+1)
                 ArraySelected=drawselected(px,py)
                 value = -1
                 redraw()
        elif key == 2424832:
             # print('this is left')
             if CaseSelected:
                 py = max(0,py-1)
                 ArraySelected=drawselected(px,py)
                 value = -1
                 redraw()


        elif key == ord("l"):
                print ('reset list possible')
                imagel,tablist=lfp(tabentert)
                redraw()
 

        # elif key == ord("a"):
        #         print( 'delete all')
        #         # delall()
        elif key == ord("L"):
            print('toggle display all possible')
            showPossible= not(showPossible)
            # imagel=lfp(imagel)
            redraw()
        elif key == ord("C"):
             print('Color')
             color=True
             # imagel=lfp(imagel)
             redraw()
        elif key == ord("c"):
              print('No Color')
              color=False
              gridColor=np.zeros(shapeRef,np.uint8)
              # imagel=lfp(imagel)
              redraw()
        elif key == ord("x"):
             print('write file for Simple Sudoku')
             f = open(cwd+"/forSS.ss", "w")
             for i in range(9):
                 if i in(3,6): 
                     f.write('-----------\n')
                 for j in  range(9):
                     if tabentert[i,j]==0:
                         f.write('.')
                     else:
                         f.write(str(tabentert[i,j]))
                     if j in(3,6): 
                         f.write('|')
                 f.write('\n')
             f.close()
                
        elif key == ord("+") :
            cplus()

                
        elif key == ord("-"):
            cmoins()
    
        elif key == ord("v"):
            cvalue()
    
        
        elif key == ord("d"):   
            cdouble()  
        elif key == ord("t"):   
            ctriple()                  
             
        
        elif key == ord("m"):                     
                 mos=True
                 print ('modify source')
                 moins=False
                 plus=False
                 visu=False
                
        elif key == ord("r"):                            
              print ('reset')
              CaseSelected=False
              tabentert=board.copy()
              image=cv2.cvtColor(ggdinit,cv2.COLOR_BGR2RGB)
              gridVisu=np.zeros(shapeRef,np.uint8)
              tabhelp=np.zeros(shapeRef,np.uint8)
              imagel=np.ones(shapeRef,np.uint8)
              imagel,tablist=lfp(tabentert)
              redraw()
        elif key == ord("n"):                            
               print ('new blank')
               CaseSelected=False
               board= np.zeros((9, 9), dtype="int")
               ggdinit=affinit(board,(255,255,0))
               tabentert=board.copy()
               image=cv2.cvtColor(ggdinit,cv2.COLOR_BGR2RGB)
               gridVisu=np.zeros(shapeRef,np.uint8)
               tabhelp=np.zeros(shapeRef,np.uint8)
               imagel=np.ones(shapeRef,np.uint8)
               imagel,tablist=lfp(tabentert)
               redraw()
        elif key == ord("s"):
            print ('save')
            #boardu,cellLocs=loadimage('cam.jpg')
            #boarddummy,cellLocs=lookForFig(stepX,stepY,puzzleImage)
            pickle.dump( (tabentert,cellLocsSave,tablist), open( cwd+"/save.p", "wb" ) )
            # pickle.dump( tablist), open( "save.p", "wb" ) )

        elif key == ord("S"):
                       print ('solve')
                       CaseSelected=False
                       showPossible=False
                       start = time.time()
                       resultSolved,solved=solvesudokuNew(board)
                       
                       #if(solvesudoku(board)):
                       end = time.time()
                       delta = end - start
                       print ("took %.2f seconds to process" % delta)
                       if resultSolved:
                           solvedF=np.zeros((9, 9), dtype="int")
                           for i in range(9):
                              for j in range(9):
                                  if solved[i,j]==tabentert[i,j]:
                                      solvedF[i,j]=0
                                  else:
                                      solvedF[i,j]=solved[i,j]
                       else:
                            print("NO SOLUTION")
                            solvedF=board
                             # print(solvedF[i,j])
                                          
                       #tabentert=board.copy()
                       # tabresrinit=board.copy()
                       # tabentert=board.copy()
                       #sprint(board)
                       ggdinit=aff(tabentert,solvedF,(255,0,0))
                       image=cv2.cvtColor(ggdinit,cv2.COLOR_BGR2RGB)
                       gridVisu=np.zeros(shapeRef,np.uint8)
                       gridpair=np.zeros(shapeRef,np.uint8)
                       redraw()



                
        elif key == ord("1") or key == ord("2") or key == ord("3")  or key == ord("4") or \
            key == ord("5") or key == ord("6") or key == ord("7") or key == ord("8") or key == ord("9") or key == ord("0") :
                value=int(chr(key))
                cchiffre(value)
      
        elif key==3014656:
           if mos:
               value=3014656
               print ('add to source',value)
           
                    
        elif key == 26:
                print('ctrl z',lastpy,lastpx)
                if lastpy>-1 and lastpx>-1:
                    
                    if lastAction=="plus":
                        print('I undo plus',px,py)
                        tabentert[lastpx,lastpy]=0
                        tablist=tablistSaved.copy()
                    if lastAction=="mos":
                            print('I undo plus',px,py)
                            tabentert[lastpx,lastpy]=0
                            tablist=tablistSaved.copy()
                            board[px,py]=0
                            tabresrinit=board.copy()
                            ggdinit=affinit(tabresrinit,(255,255,0))
                            image=cv2.cvtColor(ggdinit,cv2.COLOR_BGR2RGB)
                            tabentert=board.copy()
                            imagel,tablist=lfp(tabentert)
                            
                    if lastAction=='moins':
                        print('I undo moins',px,py)
                        # px=lastpx
                        # py=lastpy
                        moinslundo(lastpx,lastpy,numsaved)
                        errormoins=False
                        cv2.rectangle(gridMenu, (400,30), (590,60), (0,0,0), -1)
                        # cv2.putText(gridMenu,'- '+': '+chr(key) ,(450,50), cv2.FONT_HERSHEY_PLAIN,2,(240,200,180),1)
                    value = -1
                    lastpy=-1
                    lastpx=-1
                    redraw()

        elif key == ord("q") or key == ord("Q")  or quitl or cv2.waitKey(20) & 0xFF == 27 :
            print ('on quitte', quitl)
            cv2.destroyAllWindows()
            break

        elif key == ord("h"):
            print('erase help')
            tabhelp=np.zeros(shapeRef,np.uint8)
            redraw()
            
        elif key == ord("H"):
          #tempo,tablisttempo=lfp()
          hint="single"
          print('single')
          ResulHelp,tabhelp=naked_dig(tablist,(12,20,50),1)
          if not ResulHelp:
              hint="only possible"
              print("only possible")
              ResulHelp,tabhelp= onlypossible(tablist,(12,20,50))  
          if not ResulHelp:
              print("intersection removal")
              hint="intersection removal"
              ResulHelp,tabhelp= intersecr(tablist,(12,20,50))
          if not ResulHelp:
              print('double')
              hint="double"
              ResulHelp,tabhelp=naked_dig(tablist,(16,20,50),2)
          if not ResulHelp:
              print('triple')
              hint="triple"
              ResulHelp,tabhelp=naked_dig(tablist,(19,20,50),3)
          if not ResulHelp:
                  print('quadruple')
                  hint="quadruple"
                  ResulHelp,tabhelp=naked_dig(tablist,(21,20,50),4)
          if not ResulHelp:
                print('quintuple')
                hint="quintuple"
                ResulHelp,tabhelp=naked_dig(tablist,(25,20,50),5)
          if not ResulHelp:
                      print('sextuple')
                      hint="sextuple"
                      ResulHelp,tabhelp=naked_dig(tablist,(29,20,50),6)
          if not ResulHelp:
                      print('septuple')
                      hint="septuple"
                      ResulHelp,tabhelp=naked_dig(tablist,(34,20,50),7)
          if not ResulHelp:
                    print('octuple')
                    hint="octuple"
                    ResulHelp,tabhelp=naked_dig(tablist,(39,20,50),8)
          if not ResulHelp:
                    print('Locked candidate')
                    hint="Locked candidate"
                    ResulHelp,tabhelp=locked_candidate(tablist,(39,20,50),8)
          if not ResulHelp:
                        print("xwing2")
                        hint="xwing2"
                        ResulHelp,tabhelp= xwing(tablist,(12,20,50),2)
          if not ResulHelp:
                    print("xwing3")
                    hint="xwing3"
                    ResulHelp,tabhelp= xwing(tablist,(12,20,50),3)
          if not ResulHelp:
                   print("xwing4")
                   hint="xwing4"
                   ResulHelp,tabhelp= xwing(tablist,(12,20,50),4)
          if not ResulHelp:
                        print("xwing5")
                        hint="xwing5"
                        ResulHelp,tabhelp= xwing(tablist,(12,20,50),5)
          if not ResulHelp:
                        print("xywing")
                        hint="xywing"
                        ResulHelp,tabhelp= xywing(tablist,(12,20,50))
          if not ResulHelp:
                   print('excluded based on colors')    
                   hint="excluded based on colors"
                   ResulHelp,tabhelp=exBaOnCo(tablist,(12,20,50))
          if not ResulHelp:
                   for nk in range(2,6):
                       print('hidden ',nk)    
                       hint="hidden  "+str(nk)
                       ResulHelp,tabhelp=hidden_pair(tablist,nk,(12,20,50))
                       if ResulHelp:
                           break
                
          if ResulHelp:
              print('found:',hint)
              redraw()
          else:
              print('nothing found')
        
        elif key == ord("J"):
            print("xywing")
            hint="xywing"
            ResulHelp,tabhelp= xywing(tablist,(12,20,50))
            if ResulHelp:
                print('found:',hint)
                redraw()
            else:
                   print('nothing found')
    
   
        elif key == ord("w"):
            webcamV=True
        
            
        if visu:
            # valueVisu=value
            toggleDou=False
            dou=False
            toggleTri=False
            tri=False
            value=-1
            # print('visu',valueVisu,visuActive)
            visugra(valueVisu,visuActive)
          
            redraw()
            # visu=False
        

        if moins and value >0 and CaseSelected:
          #  moins=False
            moinsl(px,py,value)
            value = -1
            redraw()
            
        if plus and value >0 and CaseSelected:
            print('I add', value)
          #  plus=False
            plusl(px,py,value)
            value = -1
            redraw()
            
        if color and CaseSelected:
            gridColor=rectfrid(py,px,gridColor,(100,0,0))
            color=False
            redraw()
            
        if mos and value >0 and CaseSelected:
             # print('I add to source', value)
           #  plus=False
             if value == 3014656:
                 value = 0
             boardold=board.copy()
             board=pluslmos(px,py,value,board)
             value = -1
             resultSolved,solved=solvesudokuNew(board)
             if resultSolved:
                 solvedF=np.zeros((9, 9), dtype="int")
                 for i in range(9):
                    for j in range(9):
                        if solved[i,j]==board[i,j]:
                            solvedF[i,j]=0
                        else:
                            solvedF[i,j]=solved[i,j]
                 solvedFTrue=np.zeros((9, 9), dtype="int")
                 for i in range(9):
                   for j in range(9):
                       solvedFTrue[i,j]=solved[i,j]
                 tabresrinit=board.copy()
                 ggdinit=affinit(tabresrinit,(255,255,0))
                 image=cv2.cvtColor(ggdinit,cv2.COLOR_BGR2RGB)
                 tabentert=board.copy()
                 imagel,tablist=lfp(tabentert)
                 redraw()
             else:
                  print("NO SOLUTION")
                  board=boardold.copy()
                   # print(solvedF[i,j])
              
        if webcamV:
            cwebcam()

        
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



def rectfrid(i,j,grid,col):
    r=col[0]
    g=col[1]
    b=col[2]
    x0=int(cellLocs[j][i][0])
    x1=int(cellLocs[j][i][2])
    y0=int(cellLocs[j][i][1])
    y1=int(cellLocs[j][i][3])
    grid=cv2.rectangle(grid, (x1,y1), (x0,y0), (r,g,b), -1)   
    return grid



def combinliste(seq, k, b):
    p = []
    q=[]
    i, imax = 0, 2**len(seq)-1
    while i<=imax:
        s = []
        qq=[]
        j, jmax = 0, len(seq)-1
        while j<=jmax:
            if (i>>j)&1==1:
                if len(seq[j])>0:
                    s.append(seq[j])
                    qq.append(b[j])
            j += 1
        if len(s)==k:
            p.append(s)
            q.append(qq)
        i += 1 

    for i in range(len(p)):
        # print(i)
        resultu=[]
        #print(p[i])
        for ii in range(k):
            # print(ii,'ii')
            # print('pideii',p[i][ii])
            for jj in range(len(p[i][ii])):
                #print(jj,'jj')
                #print(result[i][ii][jj])
                if p[i][ii][jj] not in resultu:
                   resultu.append(p[i][ii][jj])
        # print('resultu',resultu)

        if len(resultu)==k:
            # print(resultu)
            # print(p[i],q[i])
            for uu in resultu:
                for jj,ll in enumerate (b):
                    if ll not in q[i]:
                        if uu in seq[jj]:
                            # print(q[i])
                            # print('unique')
                            return (True,q[i])
               
    return False,b
            
def combinliste_locked(a,n,b):
    p = []
    #for n in range(3):
    for u in range(9):
        if n in a[u]:
            p.append(b[u])
    #line
    # print('a',a)
    # print('p',p)
   
    xl=[]
    yl=[]
    if len(p)>1:
        for y in (p):
            xl.append(y[0])
            yl.append(y[1])

        xl=set(xl)
        yl=set(yl)
        # print('xl',xl)
        # print('yl',yl)
                
        if len(xl)>1:
            #print('colonne',n,p)
            sets=[(0,1,2),(3,4,5),(6,7,8)]
            for ist,lst in  enumerate(sets):
                sameset=True
                for ii in xl:
                    if ii not in lst:
                        sameset=False
                        break
                if sameset:
                    #print('GOOD COL',sets[ist],p)
                    return(True,'C',p,sets[ist])
                
        if len(yl)>1:
            #print('line',n,p)
            sets=[(0,1,2),(3,4,5),(6,7,8)]
            for ist,lst in  enumerate(sets):
                sameset=True
                for ii in yl:
                  if ii not in lst:
                      sameset=False
                      break
                if sameset:
                   #print('GOOD LINE',sets[ist],p)
                   return(True,'L',p,sets[ist])
    return(False,'X',p,p)

def isgoodnumber(table,n):
    print('run isgoodnumber')
    #define places where it is not completed
    #line
    for i in range(9):
        a=[]
        for j in range(9):
            #print((i,j),table[i][j])
            if table[i][j]==n:
                a.append((i,j))
        if len(a)!=1 :
            print('unique line',i)
            print(a)
            return False,a
    for i in range(9):
         a=[]
         for j in range(9):
             #print((i,j),table[i][j])
             if table[j][i]==n:
                 a.append((i,j))
         if len(a)!=1:
             print('unique col ',i)
             print(a)
             return False,a
             
    for i in (0,1,2):
        for j in (0,1,2):
         sectopx = 3 * i
         sectopy = 3 * j
         a=[]
         #print('sectopx:',sectopx,'sectopy:',sectopy)
         for xj in range(sectopx, sectopx+3):
                for xi in range(sectopy, sectopy+3):
           #         print((xi,xj),table[xi][xj])
                    if table[xi][xj]==n:
                        a.append((xj,xi))
         if len(a)!=1:
            print('unique carre ',i,j)
            print(a)
            return False ,a                
    return True,n

def iswrong(table,n,comment):
    if comment:
        print('run iswrong',n)
    #define places where it is not completed
    #line
    for i in range(9):
        a=[]
        for j in range(9):
            #print((i,j),table[i][j])
            if table[i][j]%10==n:
                a.append((i,j))
        if len(a)==0:
            if comment:
                print('wrong line',i)
            return True
    for i in range(9):
         a=[]
         for j in range(9):
             #print((i,j),table[i][j])
             if table[j][i]%10==n:
                 a.append((i,j))
         if len(a)==0:
             if comment:
                 print('wrong col ',i)
             return True
             
    for i in (0,1,2):
        for j in (0,1,2):
         sectopx = 3 * i
         sectopy = 3 * j
         a=[]
         #print('sectopx:',sectopx,'sectopy:',sectopy)
         for xj in range(sectopx, sectopx+3):
                for xi in range(sectopy, sectopy+3):
           #         print((xi,xj),table[xi][xj])
                    if table[xi][xj]%10==n:
                        a.append((xj,xi))
         if len(a)==0:
            if comment:
                 print('wrong carre ',i,j)
            return True   
    if comment:
        print('not wrong')
    return False

def tobenext(table,n,comment):
    if comment:
       print('run tobenext n:',n)
       print('table',table)
    #line
    for i in range(9):
        a=[]
        for j in range(9):
            # if comment:
            #     print('tobenext Line',(i,j),table[i][j])
            # if (j,i) in b:
                if table[i][j]==10+n:
                    if comment:
                       print('tobenext OK line',(j,i),table[i][j])
                    a.append((j,i))
        if len(a)==1:
            if comment:
               print('unique line',i,a[0])
            return False,a[0]
    for i in range(9):
         a=[]
         for j in range(9):
             # if comment:
             #     print('tobenext Colonne',(i,j),table[i][j])
             # if (i,j) in b:
                 # print((i,j),table[j][i])
                 if table[j][i]==10+n:
                     if comment:
                        print('tobenext OK colonne',(i,j),table[i][j])
                     a.append((i,j))
         if len(a)==1:
             if comment:
                print('unique col ',i,a[0])
             return False,a[0]
             
    for i in (0,1,2):
        for j in (0,1,2):
         sectopx = 3 * i
         sectopy = 3 * j
         a=[]
         # if comment:
         #     print('sectopx:',sectopx,'sectopy:',sectopy)
         for xj in range(sectopx, sectopx+3):
                for xi in range(sectopy, sectopy+3):
                    # if (xi,xj) in b:
           #         print((xi,xj),table[xi][xj])
                        if table[xi][xj]==10+n:
                            if comment:
                                print('tobenext OK carre',(xj,xi),table[i][j])
                            a.append((xj,xi))
         if len(a)==1:
            if comment:
               print('unique carre ',xj,xj,a[0])
            # print(a)
            return False ,a[0]              
    return True,n
               
               
def fillforone(table,n,b,comment):
    if comment:
      print('fillforone at address',b,'n',n)
      print('table',table)
    #line
    tempotable =table.copy()
    tempotable[b[1]][b[0]]=n
    i=b[1]
    for j in range(9):
        if j !=  b[0]:
          tempotable[i][j]=0
    j=b[0]
    for i in range(9):
             if i !=  b[1]:
               tempotable[i][j]=0

    sectopx = 3 * (b[0]//3)
    sectopy = 3 * (b[1]//3)

    #print('sectopx:',sectopx,'sectopy:',sectopy)
    for xj in range(sectopx, sectopx+3):
           for xi in range(sectopy, sectopy+3):
               if xi !=  b[1] and xj != b[0]:
                 tempotable[xi][xj]=0
    if comment:
       print('fillforone  result',tempotable)
    
    return tempotable

def rempliall(table,n,a,comment):
    if comment:
        print('run rempliall',n,'with seed',a)
    tempotable =table.copy()
    aa=a
    isfinish=False
    iii=0
    good=True
    while (not isfinish) and iii <5:
        iii+=1
        tempotable=fillforone(tempotable,n,aa,comment) 

        # b=lookForPossibleTabn(tempotable,n,tabentert)
        if comment:
            # print("b")
            # print(b)
    
            print(iii, 'tempotable',aa)
            print(tempotable)
        isfinish=iswrong(tempotable,n,comment)
        if not isfinish:
            if comment:
                print('rempliall not finish')
            isfinish,aa =tobenext(tempotable,n,comment)
            if not isfinish:
                if comment:
                    print('not isfinish rempliall',aa)
        else:
            if comment:
                print('rempliall impossible')
            good=False
    return tempotable,good


def lookForPossibleTabn(tab,num,arr):
    listpf=[]
    for xi in range(9):
        for yi in range(9):
            if check_row(tab,num,xi) and check_col(tab,num,yi) and not  check_cell(tab,num,xi,yi) :
                # if tabent[xi][yi] ==0:
                    if num in arr[xi,yi]:
                        listpf.append((yi,xi))
            # if check_col(tab,num,yi) :
            #     listpf.append((xi,yi))
            # if check_cell(tab,num,xi,yi):
            #     listpf.append((xi,yi))

    return listpf


def intersecr(arr,col):
    comment=False
    grid = np.zeros(shapeRef, dtype=np.uint8)
    for n in range(1,10):
        for j in (0,3,6):
             for i in (0,3,6):
                 sectopx = 3 * (j//3)
                 sectopy = 3 * (i//3)
                 # print('sectopx:',sectopx,'sectopy:',sectopy)
                 b=[]
                 for xj in range(sectopx, sectopx+3):
                     for xi in range(sectopy, sectopy+3):
                         # print(xj,xi,arr[xi,xj])
                         if n in arr[xi,xj]:
                             # print('xi:',xi,'xj:',xj)
                             b.append((xj,xi))
                 if len(b)>0:
                     xl=[]
                     yl=[]
                     for bb in b:
                        xl.append(bb[0])
                        yl.append(bb[1])
                     # print('xl',xl)
                     # print('yl',yl)
                     if len(set(yl))==1 :
                        for zz in range(9):
                             if zz not in xl:
                                 if comment:
                                     print(zz,arr[yl[0],zz])
                                 if n in arr[yl[0],zz]:
                                     b.append((zz,yl[0]))
                                     if comment:
                                          print(n)
                             
                           
                        # print(n,"only possible in carre b",b)
                                     for nri in range(len(b)):
                                        grid= rectfrid(b[nri][0],b[nri][1],grid,col)
                                     return True,grid

                     if len(set(xl))==1 :
                       for zz in range(9):
                            if zz not in yl:
                                if comment:
                                    print(zz,arr[xl[0],zz])
                                if n in arr[zz,xl[0]]:
                                    b.append((xl[0],zz))
                                    if comment:
                                      print(n)

                                    for nri in range(len(b)):
                                       grid= rectfrid(b[nri][0],b[nri][1],grid,col)
                                    return True,grid
                        
    return False,grid

def lookForHiddenPair(data,setdata,nk,comment):
    #setdata: list of figures
    #data list of possible in full set

    # calculate list of nk tuple in setdata
    p = []
    i, imax = 0, 2**len(setdata)-1
    while i<=imax:
        s = []
        j, jmax = 0, len(setdata)-1
        while j<=jmax:
            if (i>>j)&1==1:
                    s.append(setdata[j])
            j += 1
        if len(s)==nk:
            p.append(s)
        i += 1
    if comment:
         print("setdata",setdata)
         print("p",p)
    #p is list of nk tuple in setdata
    # for each tuple of p ,look for boxes with at least one figure
    # put in dictionary setn
    setn={}
    for enum,pp in enumerate(p):
        # print("pp",pp)
        setn[enum]={}
        setn[enum]['pp']=pp
        setn[enum]['data']=[]
                   
        for d in data:
            g=False
            for ss in pp:
                # if comment:
                #     print("ss",ss,"d",d)
                if ss in d[1]:
                    g=True
                    break
            if g:
                setn[enum]['data'].append(d)
    #inside setn, for each ktuple,  look for excatly nk boxes
    # put in resu dictionary 
    resu={}
    if comment:
        # print("setn",setn)
        for nn in setn:
            if len(setn[nn]['data'])>0:
                  print("pp:",setn[nn]['pp'],"data:",setn[nn]['data'])
    num=0              
    for nn in setn:
            # print('nn',nn,'len',len(setn[nn]['data']))
            if len(setn[nn]['data'])==nk:
                num+=1
                # print('nn',nn,'len',len(setn[nn]['data']))
                resu[num]={}
                resu[num]['data']=setn[nn]['pp'] #ktuple
                resu[num]['add']=[] # adresses of boxes
                resu[num]['arr']=[] # content of boxes
                
                for i in range(nk):
                     resu[num]['add'].append(setn[nn]['data'][i][0])
                     resu[num]['arr'].append(setn[nn]['data'][i][1])
                #     resu[setn[nn]['pp']]['data'].append(setn[nn]['data'][i][1])
                # if comment:
                #     print(resu)
                #     print("nn:",nn,setn[nn])
                
    #             # resu.append((setn[nn]['pp'],setn[nn]['data']))
    if comment:
          print("resu",resu)
    #     for r in resu:
    #         print('r',resu[r]['data'])
    #         print('r',resu[r]['add'])
    #         print('r',resu[r]['arr'])
            # print('r1',r[1])
    final=[]
    finalG=False
    # select nktuple and associated boxes wher no other occurence of figures outside theses boxes
    for r in resu:
        g=True
        # print('r',resu[r])
        for rr in resu[r]['data']:
            # print('rr',rr)
            for d in data:
                # print('d',d,'d0',d[0],'resu[r][add]',resu[r]['add'],'d1',d[1])
                if d[0] not in resu[r]['add']:
                    if rr in d[1]:
                        g=False
                        break
                
        if g:
            final=resu[r]
            if comment:
              print('good result',resu[r])
            finalG=True
            break
    # print('final',final)
    # look that ktuple figures are not alone in at least one box
    goodagain=False
    if finalG:
        for iarr in final['arr']:
            # print("iarr",iarr)
            if len(iarr)>nk:
                goodagain=True
                break

    
    if goodagain:
        if comment:
            print(final)
        return True,final['add']
    else:
        return False,()
        

    
def hidden_pair(arr,nk,col):
    grid = np.zeros(shapeRef, dtype=np.uint8)
    comment=False
    for i in range(0,9):
    #for i in range(4,5):
        if comment:
            print('col', i)
        data=[]
        setdata=[]
        for j in range(0,9):
            la=arr[j,i]
            for ii in la:
                if ii not in setdata:
                    setdata.append(ii)
            data.append(((i,j),la))
        if comment:
                print(data)
                print(setdata)
        result,b=lookForHiddenPair(data,setdata,nk,comment)
        if result:
            for nri in range(len(b)):
                      grid= rectfrid(b[nri][0],b[nri][1],grid,col)
            return True,grid
    for i in range(0,9):
        if comment:
            print('line', i)
        data=[]
        setdata=[]
        for j in range(0,9):
            la=arr[i,j]
            for ii in la:
                if ii not in setdata:
                    setdata.append(ii)
            data.append(((j,i),la))
        if comment:
                print(data)
                print(setdata)
        result,b=lookForHiddenPair(data,setdata,nk,comment)
        if result:
            for nri in range(len(b)):
                      grid= rectfrid(b[nri][0],b[nri][1],grid,col)
            return True,grid
    for j in (0,3,6):
          for i in (0,3,6):
             data=[]
             setdata=[]
             if comment:
                 print('carre', i,j)
             sectopx = 3 * (j//3)
             sectopy = 3 * (i//3)
             for xj in range(sectopx, sectopx+3):
                 for xi in range(sectopy, sectopy+3):
                     la=arr[xi,xj]
                     for ii in la:
                         if ii not in setdata:
                             setdata.append(ii)
                     data.append(((xj,xi),la))
             if comment:
                     print(data)
                     print(setdata)
             result,b=lookForHiddenPair(data,setdata,nk,comment)
             if result:
                for nri in range(len(b)):
                         grid= rectfrid(b[nri][0],b[nri][1],grid,col)
                return True,grid
                     # print(xj,xi,arr[xi,xj])
    return False,grid
    
def exBaOnCo(arr,col):
    grid = np.zeros(shapeRef, dtype=np.uint8)
    comment=False
    
    for n in range(1,10):
        if comment:
            print("line",'number',n)
        tempoboard =np.zeros((9, 9), dtype="int")
      
        for j in range(9):     
            for i in range(9):
                if tabentert[i,j]==n:
                    tempoboard[i,j]=n
                if n in arr[j,i]:
                    # a.append(arr[j,i])
                    tempoboard[j,i]=n+10
        if comment:
            print(tempoboard)
   
        # # seed=(4,7)
        b=lookForPossibleTabn(tempoboard,n,arr)
        if comment:
            print("b",b)
            # b=[(6,2)]
       
        for seed in b:
                tempotable,good=rempliall(tempoboard,n,seed,comment)
                # bb=lookForPossibleTabn(tempotable,n,arr)

                if not good:
                    print('value excluded',n)
                    if comment:
                         for nri in range(len(b)):
                                  grid= rectfrid(b[nri][0],b[nri][1],grid,col)

                    grid= rectfrid(seed[0],seed[1],grid,(10,50,10))
                    return True,grid
        
    return False,grid

def onlypossible(arr,col):
    grid = np.zeros(shapeRef, dtype=np.uint8)
    for n in range(1,10):
        # print("line",'number',n)
        for j in range(9):   
            b=[]
            for i in range(9):
                if n in arr[j,i]:
                    b.append((i,j))
        # print("a",a)
            if len(b)==1:
                # print(n,"only possible in line b",b)
                for nri in range(len(b)):
                    grid= rectfrid(b[nri][0],b[nri][1],grid,col)
                return True,grid
        # print("colonne",'number',n)
        for j in range(9):   
            b=[]
            for i in range(9):
                if n in arr[i,j]:
                    b.append((j,i))
        # print("a",a)
            if len(b)==1:
                # print(n,"only possible in colonne b",b)
                for nri in range(len(b)):
                    grid= rectfrid(b[nri][0],b[nri][1],grid,col)
                return True,grid
        # print("carre",'number',n)
        for j in (0,3,6):
             for i in (0,3,6):
                 sectopx = 3 * (j//3)
                 sectopy = 3 * (i//3)
                 # print('sectopx:',sectopx,'sectopy:',sectopy)
                 b=[]
                 for xj in range(sectopx, sectopx+3):
                     for xi in range(sectopy, sectopy+3):
                         # print(xj,xi,arr[xi,xj])
                         if n in arr[xi,xj]:
                             # print('xi:',xi,'xj:',xj)
                             b.append((xj,xi))
                 if len(b)==1:
                        # print(n,"only possible in carre b",b)
                        for nri in range(len(b)):
                            grid= rectfrid(b[nri][0],b[nri][1],grid,col)
                        return True,grid
    return False,grid
                         
      
def locked_candidate(arr,col,n):
    comment=False
    grid = np.zeros(shapeRef, dtype=np.uint8)
    for n in range(1,10):
        if comment:
            print("line",'number',n)
        for j in range(9):
            a=[]
            b=[]
            for i in range(9):
                a.append(arr[j,i])
                b.append((i,j))
            # print(a)
            #print(naked(a,b))
            
            Result,P,p,sets=combinliste_locked(a,n,b)
            if Result:
                # print('1')
                    # print('p result is', p)
                    # print('for line', j)
                    # print('sets=',sets)
                    sectopx = 3 * (j//3)
                    sectopy = 3 * (sets[0]//3)
                    # print('sectopx:',sectopx,'sectopy:',sectopy)
                    for xj in range(sectopx, sectopx+3):
                        for xi in range(sectopy, sectopy+3):
                            # print(arr[xj,xi],(xi,xj))
                            if n in arr[xj,xi] :
                                if (xi,xj) not in p:
                                  #print('goodr for ',(xi,xj))
                                  p.append((xi,xj))
                                  for nri in range(len(p)):
                                      grid= rectfrid(p[nri][0],p[nri][1],grid,col)
                                  return True,grid
                    # if goodr:
                    #     print(p)
                    #     for nri in range(len(p)):
                    #         grid= rectfrid(p[nri][0],p[nri][1],grid,col)
                    #     return True,grid
                #♦print(grid.shape)
                # return True,grid
        if comment:
            print("colonne",'number',n)
        for j in range(9):
             a=[]
             b=[]
             for i in range(9):
                 a.append(arr[i,j])
                 b.append((j,i))
             Result,P,p,sets=combinliste_locked(a,n,b)
             if Result:
                 # print('1')
                     # print('p result is', p)
                     # print('for colonne', j)
                     # print('sets=',sets)
                     sectopy = 3 * (j//3)
                     sectopx = 3 * (sets[0]//3)
                    # print('sectopx:',sectopx,'sectopy:',sectopy)
                     for xj in range(sectopx, sectopx+3):
                         for xi in range(sectopy, sectopy+3):
                            # print(arr[xj,xi],(xi,xj))
                             if n in arr[xj,xi] :
                                 if (xi,xj) not in p:
                                     #print('goodr for ',(xi,xj))
                                     p.append((xi,xj))
                     # if goodr:
                     #     print('goodr')
                     #     print(p)
                                     for nri in range(len(p)):
                                         grid= rectfrid(p[nri][0],p[nri][1],grid,col)
                                     return True,grid
    return False,grid

def isgood (seq):
    pos=list(itertools.permutations([0, 1,2, 3]))
    for p in pos:
        ap=p[0]
        bp=p[1]
        cp=p[2]
        dp=p[3]
        final= (seq[ap][0]== seq[bp][0]) and (seq[cp][0]==seq[dp][0]) and (seq[ap][1]== seq[cp][1]) and (seq[bp][1]==seq[dp][1])
        if final:
            return final
    return False


def combin4(seq, k):

    p = []
    i, imax = 0, 2**len(seq)-1
    while i<=imax:
        s = []
        j, jmax = 0, len(seq)-1
        while j<=jmax:
            if (i>>j)&1==1:
                if len(seq[j])>0:
                    s.append(seq[j])
            j += 1
        if len(s)==k:
            p.append(s)
        i += 1 
    return p

def compagnon(arr,b):
    # print('run compagnon')
    X=[]
    Y=[]
    C=[]
    T=[]
    x=b[0]
    y=b[1]
    sectopx = 3 * (x//3)
    sectopy = 3 * (y//3)

    #line
    for i in range(9):
        if len (arr[i,x])==2 and i not in (sectopy,sectopy+1,sectopy+2):
            T.append(((x,i),arr[i,x]))        
            Y.append(((x,i),arr[i,x]))
    #col
    for i in range(9):
        if len (arr[y,i])==2 and i not in (sectopx,sectopx+1,sectopx+2):
              X.append(((i,y),arr[y,i]))
              T.append(((i,y),arr[y,i]))
    #carre

    # print('sectopx:',sectopx,'sectopy:',sectopy)
    for xj in range(sectopx, sectopx+3):
        for xi in range(sectopy, sectopy+3):
            # if  (xi,xj) != (y, x):
            #     print((xj,xi),arr[xi,xj],(x,y))
            if len (arr[xi,xj])==2 and (xi,xj) != (y, x):
                # print('good compagnon T')
                C.append(((xj,xi),arr[xi,xj]))
                T.append(((xj,xi),arr[xi,xj]))
    # print('Ccompagnon',C)
    return X,Y,C,T


def lookforcommon(a,u):
    if a ==u:
        return False ,0,0
    if a[0]==u[0]:
        return True,u[1],a[1]
    if a[0]==u[1]:
        return True,u[0], a[1]
    if a[1]==u[0]:
        return True, u[1], a[0]
    if a[1]==u[1]:
        return True, u[0],a[0] 
    return False ,0,0
    

                         
def xywing(arr,col):

    comment=False
    if comment: print('start xywing')
    col1=(100,10,10)
    col2=(10,100,10)
    col3=(100,100,100)
    # col4=(100,10,100)
    # col5=(10,100,100)

    for i in range(0,9):
        for j in range(0,9):
    # grid = np.zeros(shapeRef, dtype=np.uint8)
    # for i in range(3,4):
    #     for j in range(1,2):
                if len (arr[j,i])==2:
                    grid = np.zeros(shapeRef, dtype=np.uint8)
                    pivot= arr[j,i]

                    if comment:
                        print('pivot',pivot,(i,j))   
                        grid= rectfrid(i,j,grid,col3)
                        # return True,grid
                    X,Y,C,T=compagnon(arr,(i,j))
                    if comment: print('recherche pince 1 en X',X)
                    #line
                    for x in X :
                        
                          pince1l=x[0]
                          pince1d=x[1]
                          # if comment:
                          #     print('pivot ',pivot,'location pince1',pince1l,'data pince 1',pince1d)
                          resokx,otherFig,candForElim=lookforcommon(pince1d,pivot)
                          if resokx:
                              if comment:
                                  print('pince1 en X, rsokx',resokx,'pivot loc', (i,j),'pivot', pivot,' pince1 loc:',pince1l, 'pince data',pince1d)
                                  print('otherFig ',otherFig,'candForElim ',candForElim)
                                  # grid= rectfrid(pince1l[0],pince1l[1],grid,col1)
                                  # return True,grid

                              if comment: print('recherche pince 2 en Y',Y)
                              for y in Y:
                                      pince2l=y[0]
                                      pince2d=y[1]
                                      if comment:
                                          print('y',y)
                                      if otherFig in pince2d:
                                          if candForElim in pince2d:
                                              candLocation=(pince1l[0],pince2l[1])
                                              if candForElim in arr[candLocation[1],candLocation[0]]:
                                                  if comment:
                                                      print('goody from x', 'y', y, 'otherfig ',otherFig )
                                                      grid= rectfrid(pince1l[0],pince1l[1],grid,col1)
                                                      grid= rectfrid(pince2l[0],pince2l[1],grid,col2)
                                                      print('candLocation',candLocation)
                                                      
                                                  grid= rectfrid(candLocation[0],candLocation[1],grid,col)
                                                  grid= rectfrid(i,j,grid,col3)
                                                  return True,grid
                              if comment: print('recherche pince 2 en carre',C)
                              for c in C:
                                         pince2l=c[0]
                                         pince2d=c[1]
                                         if comment:
                                             print('c',c,)
                                             print('otherFig ',otherFig,'candForElim ',candForElim)
                                             print('otherFig in pince2d',otherFig in pince2d)
                                             print('candForElim in pince2d',candForElim in pince2d)
                                         if otherFig in pince2d:
                                             if candForElim in pince2d:
                                                 x0x=3*(pince2l[0]//3)
                                                 candt=((x0x,pince1l[1]),(x0x+1,pince1l[1]),(x0x+2,pince1l[1]))
                                                 if comment:
                                                   print('candidate goody from carre ', 'c:', c, 'otherfig:',otherFig,"candForElim:",candForElim)
                                                   print('candt',candt)
                                                  
     
     
                                                 for cand in candt:
                                                     if candForElim in arr[cand[1],cand[0]]:
                                                         if comment:
                                                             print("cand ",cand)
                                                             grid= rectfrid(pince1l[0],pince1l[1],grid,col1)
                                                             grid= rectfrid(pince2l[0],pince2l[1],grid,col2)
                                                             # grid= rectfrid(candt[0][0],candt[0][1],grid,col4)
                                                             # grid= rectfrid(candt[1][0],candt[1][1],grid,col4)
                                                             # grid= rectfrid(candt[2][0],candt[2][1],grid,col4)
                                                         grid= rectfrid(cand[0],cand[1],grid,col)
                                                         grid= rectfrid(i,j,grid,col3)
                                                         return True,grid
                              # if comment: print('recherche pince 2 en C',C)
                                              

                    if comment: 
                             print('recherche pince 1 en Carre',C)
                    for c in C:
                        pince1l=c[0]
                        pince1d=c[1]
                        # if comment:
                        #     print('pivot ',pivot,'location pince1',pince1l,'data pince 1',pince1d)
                        resokx,otherFig,candForElim=lookforcommon(pince1d,pivot)
                        if resokx:
                            if comment:
                                print('rsokx',resokx,'pivot loc', (i,j),'pivot', pivot,' pince1 loc:',pince1l, 'pince data',pince1d)
                                print('otherFig ',otherFig,'candForElim ',candForElim)
                                # grid= rectfrid(pince1l[0],pince1l[1],grid,col1)
                                # return True,grid

                            if comment: print('recherche pince 2 en Y',Y)
                            for y in Y:
                                    pince2l=y[0]
                                    pince2d=y[1]
                                    if pince2l[0]!= pince1l[0]:
                                        # if comment:
                                        #     print('y',y)
                                        if otherFig in pince2d:
                                            if candForElim in pince2d:
                                                x0x=3*(pince2l[1]//3)
                                                candt=((pince1l[0],x0x),(pince1l[0],x0x+1),(pince1l[0],x0x+2))
                                                if comment:
                                                  print('candidate goody from carre ', 'y:', y, 'otherfig ',otherFig)
                                                  print('candt',candt)
    
    
                                                for cand in candt:
                                                    if candForElim in arr[cand[1],cand[0]]:
                                                        if comment:
                                                            print("cand ",cand)
                                                            grid= rectfrid(pince1l[0],pince1l[1],grid,col1)
                                                            grid= rectfrid(pince2l[0],pince2l[1],grid,col2)
                                                            # grid= rectfrid(candt[0][0],candt[0][1],grid,col4)
                                                            # grid= rectfrid(candt[1][0],candt[1][1],grid,col4)
                                                            # grid= rectfrid(candt[2][0],candt[2][1],grid,col4)
                                                        grid= rectfrid(cand[0],cand[1],grid,col)
                                                        grid= rectfrid(i,j,grid,col3)
                                                        return True,grid
                                                
                            if comment: print('recherche pince 2 en X',X)
                            for x in X:
                                         pince2l=x[0]
                                         pince2d=x[1]
                                         if pince2l[1]!= pince1l[1]:

                                             if otherFig in pince2d:
                                                 if candForElim in pince2d:
                                                     x0x=3*(pince2l[0]//3)
                                                     candt=((x0x,pince1l[1]),(x0x+1,pince1l[1]),(x0x+2,pince1l[1]))

                                                     if comment:
                                                       print('candidate goodx from carre ', 'x:', x, 'otherfig ',otherFig)
                                                       print('candt',candt)
                                                       # return True,grid
                                                     for cand in candt:
                                                         # print( arr[cand[1],cand[0]])
                                                         # return True ,grid
                                                         if candForElim in arr[cand[1],cand[0]]:
                                                             if comment:
                                                                 # print('goodx from carre', 'x', x, 'otherfig ',otherFig ,"candt ",candt)
                                                                 print("cand ",cand)
                                                                 grid= rectfrid(pince1l[0],pince1l[1],grid,col1)
                                                                 grid= rectfrid(pince2l[0],pince2l[1],grid,col2)
                                                                 # grid= rectfrid(candt[0][0],candt[0][1],grid,col4)
                                                                 # grid= rectfrid(candt[1][0],candt[1][1],grid,col4)
                                                                 # grid= rectfrid(candt[2][0],candt[2][1],grid,col4)
                                                             grid= rectfrid(cand[0],cand[1],grid,col)
                                                             grid= rectfrid(i,j,grid,col3)
                                                             return True,grid
                                                
                                                   
    return False ,grid
                           

def xwing(arr,col,c):
    comment=False
    grid = np.zeros(shapeRef, dtype=np.uint8)
    for n in range(1,10):
        if comment:
            print("line",'number',n)
        b=[]
        for j in range(9):
            u=[]
            for i in range(9):
                #print(arr[j,i])
                if n in arr[j,i]:
                    u.append((i,j))
            if len(u)>0 and len(u)<=c:
                # a.append((j,i))
         #       print("ok for " , j)
                # a.append(j)
                # for uu in u:
                  b.append(list(u))
        if comment:         
            print('b',b)

            
        finish,grid=xwing_verif(b,grid,arr,col,n,c)
        if finish:
            return True,grid 
        if comment:
            print("colonne",'number',n)
        b=[]
        grid = np.zeros(shapeRef, dtype=np.uint8)
        for i in range(9):
            u=[]
            for j in range(9):
                #print(arr[j,i])
                if n in arr[j,i]:
                    u.append((i,j))
            if len(u)>0 and len(u)<=c:
                # a.append((j,i))
         #       print("ok for " , j)
                # a.append(j)
                # for uu in u:
                  b.append(list(u))
        if comment:         
            print('b',b)
        finish,grid=xwing_verif(b,grid,arr,col,n,c)
        if finish:
            return True,grid 
    return False,grid




def xwing_verif(b,grid,arr,col,n,c):
    comment=False
    ps=combin4(b,c)
    if comment:
        print("ps",ps)
        for i in range(len(ps)):
            psi=ps[i]
            print("psi",psi)

    # return True,grid
    uu=[]
    for seq in ps:
        if comment: print('seq',seq,'c',c,'n',n)
        xl=[]
        yl=[]
        for i in range(len(seq)):
            # print(i,seq[i])
            for j in range(len(seq[i])):
                yl.append(seq[i][j][1])
                xl.append(seq[i][j][0])
                grid= rectfrid(seq[i][j][0],seq[i][j][1],grid,(100,0,0))
            # print(i,seq[i],xl,yl)
        if comment:
            print('xl',set(xl),'yl',set(yl))
        if len(set(xl))==c and len(set(yl))==c:
            for xi in xl:
                for yi in range(0,9): 
                    if comment: print('yi',yi,'xi',xi)
              #             print((seq[0][0]),xj)
              #            print((seq[1][0]),xj)
                    if yi not in yl:
                                # print(xj,arr[xj,seq[0][0]])
                                if n in arr[yi,xi]:
                                  if comment : print('good xwing verif',n,(xi,yi),arr[yi,xi])
                                  uu.append((xi,yi))
                                  for nri in range(len(uu)):
                                        grid= rectfrid(uu[nri][0],uu[nri][1],grid,col)
                                  return True,grid
        else:
          grid = np.zeros(shapeRef, dtype=np.uint8)
    return False,grid

        

def naked_dig(arr,col,n):
    grid = np.zeros(shapeRef, dtype=np.uint8)
    # print('line')
    for j in range(9):
        a=[]
        b=[]
        for i in range(9):
            a.append(arr[j,i])
            b.append((i,j))
        # print(a)
        #print(naked(a,b))
        
        Result,c=combinliste(a,n,b)
        if Result:
            # print('1')
            # print(c,c[0])
            for nri in range(n):
                grid= rectfrid(c[nri][0],c[nri][1],grid,col)
            #♦print(grid.shape)
            return True,grid
    # print('col')
    for j in range(9):
        a=[]
        b=[]
        for i in range(9):
            a.append(arr[i,j])
            b.append((j,i))
        Result,c=combinliste(a,n,b)
        if Result:
            # print('2')
            # print(c)
            for nri in range( n):
                grid= rectfrid(c[nri][0],c[nri][1],grid,col)
            return True,grid
    # print('square')
    for j in (0,3,6):
         for i in (0,3,6):
             a=[]
             b=[]
             sectopx = 3 * (j//3)
             sectopy = 3 * (i//3)
             #print('sectopx:',sectopx,'sectopy:',sectopy)
             for xj in range(sectopx, sectopx+3):
                 for xi in range(sectopy, sectopy+3):
                     #print('xi:',xi,'xj:',xj)
                     a.append(arr[xi,xj])
                     b.append((xj,xi))
                 # print(a)
                 #print(naked(a,b))
             
             Result,c=combinliste(a,n,b)
             if Result:
                #print('3')
                for nri in range( n):
                    grid= rectfrid(c[nri][0],c[nri][1],grid,col)
                return True,grid
    #print('4')
    return False,grid

 
def check_row(arr,num,row):
    if  all([num != arr[row][i] for i in range(9)]):
        return True
    return False

# def only_row(arr,num,row):
#     if  all([num != arr[row][i] for i in range(9)]):
#         return True
#     return False



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
def solvesudokuNew(arr):
    try:
        puzzle=convertBoard(arr)
        puzzle = Sudoku(3, 3, board=puzzle)
        solution = puzzle.solve(raising=True)
        solution=convertBoardInv(solution.board)
        return True, solution
    except:
        print ('*****************')
        print ('ERROR No solution')
        print ('*****************')
        return False, arr
    
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

def affichList(x,y,listp_,col):
    grid = np.zeros(shapeRef, dtype=np.uint8)
    r=col[0]
    g=col[1]
    b=col[2]
   
    x0=int(cellLocs[x][y][0])
    x1=int(cellLocs[x][y][2])
    y0=int(cellLocs[x][y][1])
    y1=int(cellLocs[x][y][3])
    for i in listp_:
        px=x0+int((x1-x0)/3)*((i-1)%3)+6
        py=y0+int((y1-y0)/3)*((i-1)//3)+15
        # pri+nt(i,px,py)
        cv2.putText(grid, str(i),(px,py), font, 0.5, (r, g, b), 1, cv2.LINE_AA)
     
    return grid

def visui(x,y,col):
    grid = np.zeros(shapeRef, dtype=np.uint8)
    r=col[0]
    g=col[1]
    b=col[2]
   
    x0=int(cellLocs[x][y][0])
    x1=int(cellLocs[x][y][2])
    y0=int(cellLocs[x][y][1])
    y1=int(cellLocs[x][y][3])
    grid=cv2.rectangle(grid, (x1,y1), (x0,y0), (r,g,b), -1)     
    return grid



def aff(tabref,tabaf,col):
    grid = np.zeros(shapeRef, dtype=np.uint8)
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
                    cv2.putText(grid, str(tabaf[x,y]),(px,py), font, 0.6, (b, r, g), 1, cv2.LINE_AA)
            grid=cv2.rectangle(grid, (x1,y1), (x0,y0), (255,255,255), 1)
    return grid

def affinit(tabref,col):
    grid = np.zeros(shapeRef, dtype=np.uint8)
    r=col[0]
    g=col[1]
    b=col[2]
    XX=[]
    for y in range(0,9):
       for x in range(0,9):
            x0=int(cellLocs[x][y][0])
            x1=int(cellLocs[x][y][2])
            y0=int(cellLocs[x][y][1])
            y1=int(cellLocs[x][y][3])
            if x%3==0 and y%3==0 :
                x00=x0
                x11=int(cellLocs[x][y+2][2])
           
                y00=y0
                y11=int(cellLocs[x+2][y][3])
                XX.append((x00,x11,y00,y11))
            # print(x0,x1,y0,y1)
            
            px =int((x0+x1)/2)
            py=int((y0+y1)/2)
            
            if  tabref[x,y] != 0 :
                    cv2.putText(grid, str(tabref[x,y]),(px,py), font, 0.8, (b, g, r), 2, cv2.LINE_AA)
            grid=cv2.rectangle(grid, (x1,y1), (x0,y0), (80,80,80), 1)
    for l  in XX:
            x0=l[0]
            y0=l[2]
            x1=l[1]
            y1=l[3]
            grid=cv2.rectangle(grid, (x1,y1), (x0,y0), (255,255,255), 4)
    # print(XX)
    
    return grid

def lookForPossible(xi,yi,tabentert):
    listpf=[]
    for num in range(1,10):
        if check_row(tabentert,num,xi) and check_col(tabentert,num,yi) and not check_cell(tabentert,num,xi,yi):
             listpf.append(num)
    return listpf




def lookForPxPy(xi,yi):
    # global CaseSelected
    ArraySelected = np.zeros(shapeRef, dtype=np.uint8)
    color=(255,255,0)
    thickness=4
    CaseSelected=False
    px=-1
    py=-1
    foundy=False
    foundx=False
    for x in range(0,9):
          x0=int(cellLocs[0][x][0])
          x1=int(cellLocs[0][x][2])
          if xi >=x0 and xi <=x1:
              py=x
              foundy = True
              #print('good',py)
              break

    for y in range(0,9):      
          y0=int(cellLocs[y][0][1])
          y1=int(cellLocs[y][0][3])
          # print(y,y0,y1,yi)
          if yi-101 >=y0 and yi-101 <=y1:
              px=y
              #print('good',px)
              foundx = True
              break
    #      # print(x0,x1,y0,y1)
    if foundx and foundy: 
        CaseSelected=True
        start_point = (cellLocs[px][py][0], cellLocs[px][py][1]) 
        end_point = (cellLocs[px][py][2], cellLocs[px][py][3]) 
        # print("start_point",start_point)
        # print("end_point",end_point)
        ArraySelected = cv2.rectangle(ArraySelected, start_point, end_point, color, thickness) 
        # print("write jpeg",CaseSelected)
        # cv2.imwrite('a.jpg',ArraySelected)
        
    return px,py,ArraySelected,CaseSelected

def lookForFigSeed(stepX,stepY):    
    # loop over the grid locations
    cellLocs=[]
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
        cellLocs.append(row)
    return cellLocs
    

def lookForFig(stepX,stepY,puzzleImage):
    # loop over the grid locations
    cellLocs=[]
    board=np.zeros((9, 9), dtype="int")
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


def convertBoard(board):
    target=[]
    for j in range(9):
        line=[]
        for i in range(9):
            line.append(board[j][i])
        target.append(line)
    return target

def convertBoardInv(board):
    target=np.zeros((9, 9), dtype="int")
    for j in range(9):
        for i in range(9):
            if board[i][j]==None:
                target[i,j]=0
            else: 
                target[i,j]=board[i][j]
    return target


def loadimage(imageToLoad):
    global cellLocs
    print("[INFO] loading input image...",imageToLoad)
    image = cv2.imread(cwd+'/'+imageToLoad)
    
    # image =rotate_image(image,90)
    
    aspectratio=image.shape[0]/image.shape[1]
    newheight=int(finalwidth*aspectratio)
    
    image=cv2.resize(image, (finalwidth, newheight))
    
    # cv2.imshow("image", image)
    
    (puzzleImage, warped) = find_puzzle(image, debug=debug)
    
    aspectratio=warped.shape[0]/warped.shape[1]
    newheight=int(finalwidth*aspectratio)
    warped=cv2.resize(warped, (finalwidth, newheight))
    puzzleImage=cv2.resize(puzzleImage, (finalwidth, newheight))
    image=cv2.resize(image, (shapeRef[0],shapeRef[1]))
    
    
    # cv2.imshow("puzzleImage", puzzleImage)
    # cv2.imshow("warped", warped)
    # cv2.imshow("image", image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()  
    

    
    # initialize our 9x9 Sudoku board
    board = np.zeros((9, 9), dtype="int")
    # a Sudoku puzzle is a 9x9 grid (81 individual cells), so we can
    # infer the location of each cell by dividing the warped image
    # into a 9x9 grid
    stepX = warped.shape[1] // 9
    stepY = warped.shape[0] // 9
    # print(warped.shape)
    # print(image.shape)
    # initialize a list to store the (x, y)-coordinates of each cell
    # location
   # cellLocs = []

    board,cellLocs=lookForFig(stepX,stepY,puzzleImage)
    tabentert=board.copy()
    imagel,tablist=lfp(tabentert)
    pickle.dump( (board,cellLocs,tablist), open( cwd+"/save.p", "wb" ) )
    return board,cellLocs

#Radomseed=False


if Radomseed :
    seed=random.randrange(1000)
    # shapeRef=(596, 600, 3)
    shapeRef=(680, 600, 3)
    puzzle= Sudoku(3, 3,seed=seed).difficulty(0.4)
    board= convertBoardInv(puzzle.board)
    tabentert=board.copy()
    stepX = shapeRef[1] // 9
    stepY = shapeRef[0] // 9
    # print(stepX,stepY)
   # cellLocs = []
    cellLocs=lookForFigSeed(stepX,stepY)
    imagel,tablist=lfp(tabentert)


elif readJPG:
    shapeRef=(680, 600, 3)
    board,cellLocs=loadimage(imageToLoad)
    # shapeRef=(680, 600, 3)
    tabentert=board.copy()
    imagel,tablist=lfp(tabentert)
    stepX = shapeRef[1] // 9
    stepY = shapeRef[0] // 9
        # (board,cellLocs,tablist) = pickle.load( open( "save.p", "rb" ) )
        # # shapeRef=(631, 600, 3)
        # shapeRef=(680, 600, 3)
        # stepX = shapeRef[1] // 9
        # stepY = shapeRef[0] // 9
        # # imagel=lfplist()
else:
    try:
        (board,cellLocs,tablist) = pickle.load( open( cwd+"/save.p", "rb" ) )
    except:
            (board,cellLocs) = pickle.load( open( cwd+"/save.p", "rb" ) )
            shapeRef=(680, 600, 3)
            tabentert=board.copy()
            imagel,tablist=lfp(tabentert)
            
    # shapeRef=(631, 600, 3)
    shapeRef=(680, 600, 3)
    stepX = shapeRef[1] // 9
    stepY = shapeRef[0] // 9
  

ArraySelected =np.zeros(shapeRef, dtype=np.uint8)


tabresrinit=board.copy()
tabentert=board.copy()
ggdinit=affinit(tabresrinit,(255,255,0))
# puzzle=convertBoard(board)
# puzzle = Sudoku(3, 3, board=puzzle)
resultsolved,solved=solvesudokuNew(board)
solvedFTrue=np.zeros((9, 9), dtype="int")
for i in range(9):
   for j in range(9):
       solvedFTrue[i,j]=solved[i,j]

cellLocsSave =cellLocs


loop(board)

if solve:
    if(solvesudoku(board)):
        ggd=aff(tabresrinit,board,(255,255,0))
    
    else:
        ggd=ggdinit
        print("There is no solution")

