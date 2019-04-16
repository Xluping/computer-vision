import cv2
import numpy as np
import sys


def main():

    if len(sys.argv) == 3:
        img1 = cv2.imread(sys.argv[1])
        img2 = cv2.imread(sys.argv[2])
    else:
        cap = cv2.VideoCapture(0)
        for i in range(0, 15):
            retval1, img1 = cap.read()
            retval2, img2 = cap.read()
        if retval1 and retval2:
            cv2.imwrite("cap1.jpg", img1)
            cv2.imwrite("cap2.jpg", img2)

    comb = np.concatenate((img1, img2), axis=1)

    print("PRESS \n'H' for help. \n'q' to quit.\n")
    k = input()
    while k != 'q':
        if k == 'h':
            n = input("The variance of Gaussian (scale):")
            neighbourhood = input("The neighbourhood size :")
            k = input("Coefficient of trace[0, 0.5]:")
            threshold = input("Threshold:")
            print("wait...")
            output = harris(comb, n, neighbourhood, k, threshold)
            cv2.namedWindow("test_windown", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("test_windown", output)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if k == 'm':
            output = featureVector(img1, img2)
            cv2.namedWindow("test_windown", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("test_windown", output)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if k == 'l':
            output = betterLocalization(comb)
            cv2.namedWindow("test_windown", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("test_windown", output)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if k == 'H':
            print("'h': harris corner detection algorithm.")
            print("'l': Obtain a better localization of each corner.")
            print("'m': Compute a feature vector for each corner were detected.\n")
        print("PRESS \n'H' for help. \n'q' to quit.\n")
        k = input()



def harris(img, n, neighbourhood, k, threshold):
    n = int(n)
    neighbourhood = int(neighbourhood)
    k = float(k)
    threshold = int(threshold)
    copy = img.copy()
    rList = []
    h_img = img.shape[0]
    w_img = img.shape[1]
    offset = int(neighbourhood / 2)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.float32(img)
    ker = np.ones((n, n), np.float32) / (n * n)
    img = cv2.filter2D(img, -1, ker)
    dy, dx = np.gradient(img)
    Ixx = dx ** 2
    Ixy = dy * dx
    Iyy = dy ** 2

    for y in range(offset, h_img - offset):
        for x in range(offset, w_img - offset):
            wIxx = Ixx[y - offset : y + offset + 1, x - offset : x + offset + 1]
            wIxy = Ixy[y - offset : y + offset + 1, x - offset : x + offset + 1]
            wIyy = Iyy[y - offset : y + offset + 1, x - offset : x + offset + 1]
            Sxx = wIxx.sum()
            Sxy = wIxy.sum()
            Syy = wIyy.sum()
            det = (Sxx * Syy) - (Sxy ** 2)
            trace = Sxx + Syy
            r = det - k *(trace ** 2)
            rList.append([x, y, r])
            if r > threshold:
                copy.itemset((y, x, 0), 0)
                copy.itemset((y, x, 1), 0)
                copy.itemset((y, x, 2), 255)
                cv2.rectangle(copy, (x + 10, y + 10), (x - 10, y - 10), (255, 0, 0), 1)
    return copy



def featureVector(img1, img2):

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)

    matches = sorted(matches, key = lambda x:x.distance)
    l1 = []
    l2 = []
    for m in matches:
        (x1, y1) = kp1[m.queryIdx].pt
        (x2, y2) = kp2[m.trainIdx].pt
        l1.append((x1, y1))
        l2.append((x2, y2))
    for i in range(0, 100):
        p1 = l1[i]
        p2 = l2[i]
        cv2.putText(img1, str(i), (int(p1[0]), int(p1[1])),  cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2)
        cv2.putText(img2, str(i), (int(p2[0]), int(p2[1])),  cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2)
    output = np.concatenate((img1, img2), axis=1)
    return output


def betterLocalization(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    dst = cv2.dilate(dst,None)
    ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)

    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
    output = np.hstack((centroids,corners))
    output = np.int0(output)
    img[output[:,1],output[:,0]]=[255,0,0]
    img[output[:,3],output[:,2]]=[0,255,0]
    return img


main()