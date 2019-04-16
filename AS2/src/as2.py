# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 21:28:54 2018

@author: vismaya
"""

import cv2
import numpy as np
import sys
from skimage.exposure import rescale_intensity

def nothing(b):
    pass

def smoothFunction(val):
    if val > 0:
        blur = cv2.blur(gray, (val, val))
        cv2.imshow('test', blur)
        current = blur


def ownSmoothFunction(val):
    if val > 0:
        convoleOutput = ownConvole(val, gray)
        current = convoleOutput
        cv2.imshow('test',convoleOutput)


def ownConvole(image):
        kernel = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))

        (iH, iW) = image.shape[:2]
        (kH, kW) = kernel.shape[:2]

        pad = (kW - 1) // 2
        image = cv2.copyMakeBorder(image, pad, pad, pad, pad,cv2.BORDER_REPLICATE)
        output = np.zeros((iH, iW), dtype="float32")

        for y in np.arange(pad, iH + pad):
            for x in np.arange(pad, iW + pad):

                roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
                k = (roi * kernel).sum()
                output[y - pad, x - pad] = k

        output = rescale_intensity(output, in_range=(0, 255))
        #output = cv2.normalize(output,(0,255),cv2.NORM_MINMAX)
        output = (output * 255).astype("uint8")
        return output

def rotate(val):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, val, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h))
    current = rotated
    cv2.imshow('test', rotated)
    return rotated


# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required = True, help = "Path to the image")
# args = vars(ap.parse_args())
#
# image = cv2.imread(args["image"])

c=0
video = 0

if len(sys.argv) == 2:
    filename = sys.argv[1]
    image = cv2.imread(filename)
    current = image
    cv2.imshow('test', image)

elif len(sys.argv) < 2:
    cap = cv2.VideoCapture(0)
    video = 1

cv2.namedWindow('test',cv2.WINDOW_NORMAL)

while(video == 0):

    k = cv2.waitKey(1)

    if k == ord('i') : # wait for 'i' key to reload
        current = image
        cv2.imshow('test',image)

    elif k == ord('w'):
        cv2.imwrite('E:\CV\cs512-f18-vismayaveeramanju-kalyan\AS2\data\out.png',current)

    elif k == ord('g'):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        current = gray
        cv2.imshow('test', current)

    elif k == ord('G'):
        gray = 0.01 * image[:, :, 2] + 0.813 * image[:, :, 1] + 0.177 * image[:, :, 0]
        cv2.normalize(gray, 0, 255, cv2.NORM_MINMAX)
        gray_img = gray.astype(np.uint8)
        current = gray_img
        cv2.imshow('test', current)


    elif k == ord('c') :
            b, g, r = cv2.split(image)

            if c == 0:
                # cv2.imshow('test',r)
                current = r
                cv2.imshow('test', current)
                # orig = ord('c')
                c = c + 1

            elif c == 1:
                # cv2.imshow('test',b)
                current = b
                cv2.imshow('test', current)
                # orig = ord('c')

                c = c + 1

            elif c == 2:
                # cv2.imshow('test',g)
                current = g
                cv2.imshow('test', current)
                # orig = ord('c')

                c = 0

    elif k == ord('s'):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.createTrackbar('Smoothing','test', 0 , 15, smoothFunction)

    elif k == ord('S'):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        convoleOutput = ownConvole(gray)
        cv2.imshow('test', convoleOutput)
        current = convoleOutput

    elif k == ord('d'):
        scale_percent = 50  # percent of original size
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)

        resized = cv2.resize(image, dim)
        print('resized image', resized.shape)
        cv2.imshow('test', resized)
        current = resized

    elif k == ord('D'):
        scale_percent = 50  # percent of original size
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        smoothed = cv2.GaussianBlur(image,(7,7),0)
        resized = cv2.resize(smoothed, dim)
        print('resized image', resized.shape)
        cv2.imshow('test', resized)
        current = resized

    elif k == ord('x'):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        x_derivatve = cv2.filter2D(gray, -1, kernel)
        cv2.normalize(x_derivatve, 0, 255, cv2.NORM_MINMAX)
        current = x_derivatve
        cv2.imshow('test', current)


    elif k == ord('y'):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
        y_derivatve = cv2.filter2D(gray, -1, kernel)
        cv2.normalize(y_derivatve, 0, 255, cv2.NORM_MINMAX)
        current = y_derivatve
        cv2.imshow('test', current)

    elif k == ord('m'):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        x_derivatve = cv2.filter2D(gray,cv2.CV_64F , kernel_x)
        cv2.normalize( x_derivatve, 0, 255, cv2.NORM_MINMAX)
       # x_derivatve = rescale_intensity(x_derivatve, in_range=(0, 255))

        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        y_derivatve = cv2.filter2D(gray, cv2.CV_64F , kernel_y)
        cv2.normalize( y_derivatve, 0, 255, cv2.NORM_MINMAX)
        # y_derivatve = rescale_intensity(y_derivatve, in_range=(0, 255))

        # sum_xy = x_derivatve**2 + x_derivatve**2
        # mag = np.sqrt(sum_xy)
        # print("mag",mag)

        magni = cv2.magnitude(x_derivatve,y_derivatve)
        cv2.normalize(magni, 0, 255, cv2.NORM_MINMAX)
        print("\n\n magni",magni)
        current = magni
        cv2.imshow('test',magni)

    elif k == ord('p'):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        x_derivatve = cv2.filter2D(gray, cv2.CV_64F, kernel_x)
        cv2.normalize(x_derivatve, 0, 255, cv2.NORM_MINMAX)

        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        y_derivatve = cv2.filter2D(gray, cv2.CV_64F, kernel_y)
        cv2.normalize(y_derivatve, 0, 255, cv2.NORM_MINMAX)
        gradientVectorsMag, gradientVectorsAngle = cv2.cartToPolar(x_derivatve, y_derivatve)
        for i in range(0, image.shape[0]):
            for j in range(0, image.shape[1]):
                if gradientVectorsMag[i, j] > 0:
                    cv2.arrowedLine(gray, (j, i), ((j), (i)), (0, 255, 255))
        cv2.imshow('test', gray)
        current= gray

    elif k == ord('r'):
       cv2.createTrackbar('Rotation','test', 0, 360, rotate)

    elif k == ord('h'):
        print("The program takes the image from the AS2\data\car.jpeg as the argument to run. Its written using python 3. To run use the command \"python as2.py ...\AS2\data\car.jpeg\" \n")
        print("or just run python as2.py to take the image from web cam.")
        print("press \'i\' to reload the original image \n")
        print("\'w\' to save the current image \n" )
        print("\'g\' to convert the image into grayscale\n" )
        print("\'G\' to convert the image to grayscale using own implementation \n" )
        print("\'c\' to cycle through the color channels\n" )
        print("\'s\' to convert the image to grayscale and smooth it. Use the trackbar to change the kernel size \n" )
        print("\'S\' to convert the image to grayscale and smooth it using own convolution. Use the trackbar to change the kernel size  \n" )
        print("\'d\' to down sample the image by 2 factor.\n")
        print("\'D\' to down sample the image by 2 factor and smooth the image.\n")
        print("\'x\' to perform convolution with x derivative.\n")
        print("\'y\' to perform convolution with y derivative.\n")
        print("\'m\' performs gradient and normalizes and shows the magnitude of the vector \n")
        print("\'p\' plots the gradient vector of the image.\n")
        print("\'r\' converts the image to gray scale and performs rotation.\n")
        print("\'h\' help\n")


    if k == ord('q'):         # wait for ESC key to exit
        break

#video

if video == 1:
    cap = cv2.VideoCapture(0)
    opt = -1
    c = -1
    while True:
        ret, image = cap.read()
        # image = sample
        # cv2.imshow("window1",sample)
        k = cv2.waitKey(1)

        if k == -1:
            pass
        elif k == ord('i'):
            opt = ord('i')

        elif k == ord('w'):
            opt = ord('w')

        elif k == ord('g'):
            opt = ord('g')

        elif k == ord('G'):
            opt = ord('G')

        elif k == ord('c'):
            opt = ord('c')
            if c < 2:
                c = c + 1
            else:
                c = 0

        elif k == ord('s'):
            opt = ord('s')

            cv2.createTrackbar("smoothing",'test', 0, 100, nothing)

        elif k == ord('S'):
            opt = ord('S')

        elif k == ord('d'):
            opt = ord('d')

        elif k == ord('D'):
            opt = ord('D')

        elif k == ord('x'):
            opt = ord('x')

        elif k == ord('y'):
            opt = ord('y')

        elif k == ord('m'):
            opt = ord('m')

        elif k == ord('p'):
            opt = ord('p')

        elif k == ord('r'):
            opt = ord('r')
            cv2.createTrackbar("Rotation",'test', 0, 360, nothing)

        elif k == ord('h'):
            opt = ord('h')

        if k == -1 or opt == ord('i'):
            cv2.imshow('test',image)
            current = image
        if ord('g') == opt:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imshow('test', gray)
            current = image

        elif ord('i') == opt:
            cv2.imshow('test', image)
            current = image

        elif ord('w') == opt:
            cv2.imwrite('D:\CV\cs512-f18-vismayaveeramanju-kalyan\AS2\data\out.png', current)

        elif ord('G') == opt:
            gray = 0.01 * image[:, :, 2] + 0.813 * image[:, :, 1] + 0.177 * image[:, :, 0]
            cv2.normalize(gray, 0, 255, cv2.NORM_MINMAX)
            gray_img = gray.astype(np.uint8)
            current = gray_img
            cv2.imshow('test', current)

        elif ord('c') == opt:
            split = cv2.split(image)
            current = split[c]
            cv2.imshow('test', split[c])

        elif ord('s') == opt:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            trackbarPos = cv2.getTrackbarPos("smoothing", 'test')
            if trackbarPos > 0:
                basic_blur = cv2.blur(gray, (trackbarPos, trackbarPos))
            else:
                basic_blur = cv2.blur(gray, (5, 5))
            cv2.imshow('test', basic_blur)
            current = basic_blur


        elif ord('S') == opt:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            convoleOutput = ownConvole(gray)
            cv2.imshow('test', convoleOutput)
            current = convoleOutput

        elif ord('d') == opt:
            downsample= cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow('test', downsample)
            current = downsample
            # print(sample.shape)
            # print(downsample.shape)

        elif ord('D') == opt:
            Smoothed = cv2.GaussianBlur(image, (7, 7), 0)
            downsample = cv2.resize(Smoothed, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow('test', downsample)
            current = downsample
            print(image.shape)
            print(downsample.shape)

        elif ord('x') == opt:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            x_derivatve = cv2.filter2D(gray, cv2.CV_64F	, kernel)
            cv2.normalize(x_derivatve,(0,255),cv2.NORM_MINMAX)
            cv2.imshow('test', x_derivatve)
            current =x_derivatve


        elif ord('y') == opt:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            kernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
            y_derivatve = cv2.filter2D(gray, cv2.CV_64F, kernel)
            cv2.normalize(y_derivatve, (0, 255), cv2.NORM_MINMAX)
            cv2.imshow('test', y_derivatve)
            current = y_derivatve

        elif ord('m') == opt:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            x_derivatve = cv2.filter2D(gray, cv2.CV_64F, kernel_x)
            cv2.normalize(x_derivatve, 0, 255, cv2.NORM_MINMAX)

            kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
            y_derivatve = cv2.filter2D(gray, cv2.CV_64F, kernel_y)
            cv2.normalize(y_derivatve, 0, 255, cv2.NORM_MINMAX)

            magni = cv2.magnitude(x_derivatve, y_derivatve)
            cv2.normalize(magni, 0, 255, cv2.NORM_MINMAX)
            print("\n\n magni", magni)
            cv2.imshow('test', magni)
            current = magni

        elif ord('p') == opt:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            x_derivatve = cv2.filter2D(gray, cv2.CV_64F, kernel_x)
            cv2.normalize(x_derivatve, 0, 255, cv2.NORM_MINMAX)

            kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
            y_derivatve = cv2.filter2D(gray, cv2.CV_64F, kernel_y)
            cv2.normalize(y_derivatve, 0, 255, cv2.NORM_MINMAX)

            vectorsMag, vectorsAngle = cv2.cartToPolar(x_derivatve, y_derivatve)
            for i in range(0, image.shape[0]):
                for j in range(0, image.shape[1]):
                    if vectorsMag[i, j] > 0:
                        cv2.arrowedLine(gray, (j, i), ((j), (i)), (0, 255, 255))
            cv2.imshow('test', gray)
            current = gray

        elif ord('r') == opt:
            Rotate = cv2.getRotationMatrix2D((image.shape[1]/2, image.shape[0]/2), cv2.getTrackbarPos("Rotation", 'test'), 1)
            rotatedRes = cv2.warpAffine(image, Rotate, (image.shape[0], image.shape[1]))
            cv2.imshow('test', rotatedRes)
            current = rotatedRes

        elif  k == ord('h'):
            print(
                "The program takes the image from the AS2\data\car.jpeg as the argument to run. Its written using python 3. To run use the command \"python as2.py ...\AS2\data\car.jpeg\" \n")
            print("or just run python as2.py to take the image from web cam.")
            print("press \'i\' to reload the original image \n")
            print("\'w\' to save the current image \n")
            print("\'g\' to convert the image into grayscale\n")
            print("\'G\' to convert the image to grayscale using own implementation \n")
            print("\'c\' to cycle through the color channels\n")
            print(
                "\'s\' to convert the image to grayscale and smooth it. Use the trackbar to change the kernel size \n")
            print(
                "\'S\' to convert the image to grayscale and smooth it using own convolution. Use the trackbar to change the kernel size  \n")
            print("\'d\' to down sample the image by 2 factor.\n")
            print("\'D\' to down sample the image by 2 factor and smooth the image.\n")
            print("\'x\' to perform convolution with x derivative.\n")
            print("\'y\' to perform convolution with y derivative.\n")
            print("\'m\' performs gradient and normalizes and shows the magnitude of the vector \n")
            print("\'p\' plots the gradient vector of the image.\n")
            print("\'r\' converts the image to gray scale and performs rotation.\n")
            print("\'h\' help\n")

        elif k == 27:
            cap.release()
            break

cv2.destroyAllWindows()