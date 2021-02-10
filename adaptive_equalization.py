# USAGE
# python adaptive_equalization.py --image images/boston.png


from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import numpy as np
import imutils
import cv2
import pytesseract


debug=True

finalwidth=600

# load the input image from disk and convert it to grayscale
print("[INFO] loading input image...")
image = cv2.imread('s1.jpg')

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
        digit = extract_digit(cell, debug=False)
		# verify that the digit is not empty
        if digit is not None:
			# resize the cell to 28x28 pixels and then prepare the
			# cell for classification
            roi = cv2.resize(digit, (128, 128))
            roi=255-roi
            np.putmask(roi,roi > 128,255)
            np.putmask(roi,roi < 129,0)
            roi1=roi.copy()
            # print(roi.shape)
            # print(np.unique(roi))
            kernel = np.ones((3,3),np.uint8)
            roi = cv2.dilate(roi,kernel,iterations = 1)
            kernel = np.ones((4,4),np.uint8)
            roi = cv2.erode(roi,kernel,iterations = 1)
            boxes = pytesseract.image_to_boxes(roi,config='--psm 10 --oem 3 -c tessedit_char_whitelist=123456789')
            # roi = roi.astype("float") / 255.0
            boxes=boxes.split(' ')
            if True:
                print(boxes)
                cv2.imshow("roi", roi)
                cv2.imshow("roi1", roi1)
                # cv2.imshow("Digit", roi)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        # # print(boxes,boxes[0])

            try:
                restext=int(boxes[0])
            except:
                # print(boxes)
                restext=0
            if restext in [1,2,3,4,5,6,7,8,9]:
                print(restext)
            # roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
			# classify the digit and update the Sudoku board with the
			# prediction
            # pred = model.predict(roi).argmax(axis=1)[0]
            board[y, x] = restext
	# add the row to our cell locations
    cellLocs.append(row)

print(board)
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
cv2.imshow("warped", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()