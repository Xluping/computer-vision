import numpy as np
import cv2
import glob

PATTERN_SIZE = (9, 6)
SQUARE_SIZE = 1.0 

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9*6,3), np.float32)
objp[:, :2] = np.indices(PATTERN_SIZE).T.reshape(-1, 2)
objp *= SQUARE_SIZE

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
correspondences = []

images = glob.glob('../data/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # If found, add object points, image points (after refining them)
    if ret:

       
        corners_exact = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        corners_exact = corners_exact.reshape(-1, 2)

        if corners_exact.shape[0] == objp.shape[0] :
            imgpoints.append(corners_exact)
            objpoints.append(objp[:,:-1])
            assert corners_exact.shape == objp[:, :-1].shape, "mismatch shape corners and objp[:,:-1]"
            correspondences.append([corners_exact.astype(np.int), objp[:, :-1].astype(np.int)])

        # Write the points to the file
        file = open(fname + ".txt", "w")
        for i,j in zip(objp[:,:-1], corners_exact.reshape(-1,2)):
            file.write(str(i[0]) + ' ' + str(i[1]) +  ' ' + str(j[0]) + ' ' + str(j[1]) + '\n')
        file.close()

        # print("correspondences",len(correspondences))
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners_exact,ret)
        cv2.imshow('img',img)
        cv2.imwrite(fname + ".jpg", img)
        cv2.waitKey(500)

cv2.destroyAllWindows()