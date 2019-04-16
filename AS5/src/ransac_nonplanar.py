import cv2
import numpy as np
import sys
import random
import math

def main():

    file_name = sys.argv[1]
    object_point, image_point = [], []
    f = open(file_name)
    file_data = f.readlines()
    for line in file_data:
        points = line.split()
        object_point.append([float(p) for p in points[:3]])
        image_point.append([float(p) for p in points[3:]])


    configname = sys.argv[2]
    with open(configname, 'r') as conf:
        prob = float(conf.readline().split()[0])
        kmax = int(conf.readline().split()[0])
        nmin = int(conf.readline().split()[0])
        nmax = int(conf.readline().split()[0])
        w = float(conf.readline().split()[0])
    # Using RANSAC to remove outliers


    k = kmax
    np.random.seed(0)
    count = 0
    inlinerNum = 0
    M = None

    # Finding A
    a = []
    zero = np.zeros(4)
    for i, j in zip(object_point, image_point):
        pi = np.array(i)
        pi = np.concatenate([pi, [1]])
        xipi = j[0] * pi
        yipi = j[1] * pi
        a.append(np.concatenate([pi, zero, -xipi]))
        a.append(np.concatenate([zero, pi, -yipi]))


    M_inter = []
    u, s, v = np.linalg.svd(a, full_matrices = True)
    M_inter = v[-1].reshape(3, 4)

    m1 = M_inter[0][:4]
    m2 = M_inter[1][:4]
    m3 = M_inter[2][:4]
    d = []
    for i, j in zip(object_point, image_point):
        xi = j[0]
        yi = j[1]
        pi = np.array(i)
        # pi = np.concatenate([pi, [1]])
        pi = np.append(pi, 1)
        exi = (m1.T.dot(pi)) / (m3.T.dot(pi))
        eyi = (m2.T.dot(pi)) / (m3.T.dot(pi))
        di = np.sqrt(((xi - exi) ** 2 + (yi - eyi) ** 2))
        d.append(di)
    

    # To estimate t use median distance from model
    medianDistance = np.median(d) 
    t = 1.5 * medianDistance
    n = random.randint(nmin, nmax)


    for count in range(kmax):
        
        # Selecting random points of size n with replacement
        index = np.random.choice(len(object_point), n)
        ranOp, ranIp = np.array(object_point)[index], np.array(image_point)[index]
       
        A_i = []
        zero = np.zeros(4)
        for i, j in zip(ranOp, ranIp):
            pi = np.array(i)
            pi = np.concatenate([pi, [1]])
            xipi = j[0] * pi
            yipi = j[1] * pi
            A_i.append(np.concatenate([pi, zero, -xipi]))
            A_i.append(np.concatenate([zero, pi, -yipi]))

        # Fit the model to these points
       
        M_i = []
        u, d, v = np.linalg.svd(A_i, full_matrices = True)
        # print("v",v)
        # Column of v belonging to zero singular value
        # Reshape and transpose can be used interchangably
        M_i = v[-1].reshape(3, 4)
        # print("M", M)

        m1 = M_i[0][:4]
        m2 = M_i[1][:4]
        m3 = M_i[2][:4]
        d_i = []
        for i, j in zip(object_point, image_point):
            xi = j[0]
            yi = j[1]
            pi = np.array(i)
            # pi = np.concatenate([pi, [1]])
            pi = np.append(pi, 1)
            exi = (m1.T.dot(pi)) / (m3.T.dot(pi))
            eyi = (m2.T.dot(pi)) / (m3.T.dot(pi))
            di = np.sqrt(((xi - exi) ** 2 + (yi - eyi) ** 2))
            d_i.append(di)


        # Recompute the model if atleast d inliers

        inliner = []
        for i, d_i in enumerate(d_i):
            if d_i < t:
                inliner.append(i)
        if len(inliner) >= inlinerNum:
            inlinerNum = len(inliner)
            inlinerOp, inlinerIp = np.array(object_point)[inliner], np.array(image_point)[inliner]
           
            A_i = []
            zero = np.zeros(4)
            for i, j in zip(ranOp, ranIp):
                pi = np.array(i)
                pi = np.concatenate([pi, [1]])
                xipi = j[0] * pi
                yipi = j[1] * pi
                A_i.append(np.concatenate([pi, zero, -xipi]))
                A_i.append(np.concatenate([zero, pi, -yipi]))


            M = []
            u, d, v = np.linalg.svd(A_i, full_matrices = True)
            # print("v",v)
            # Column of v belonging to zero singular value
            # Reshape and transpose can be used interchangably
            M = v[-1].reshape(3, 4)
            # print("M", M)


        # Update w,k every iteration but set upper bound for k
        if not (w == 0 ):
            w = float(len(inliner))/float(len(image_point))
            k = float(math.log(1 - prob)) / np.absolute(math.log(1 - (w ** n)))
        




    # Finding P from M break M to knowns a and b
    a1 = M[0][:3].T
    a2 = M[1][:3].T
    a3 = M[2][:3].T
    # print("a1", a1)
    # print("a2", a2)
    # print("a3", a3)

    b = []
    for i in range(len(M)):
        b.append(M[i][3])
    b = np.reshape(b, (3, 1))
    # print("b", b)

    np.set_printoptions(formatter={'float': "{0:.9f}".format})

    # Compute the value of p
    magP = 1 / np.linalg.norm(a3.T) 

    # compute u0, v0
    u0 = magP ** 2 * (a1.T.dot(a3))
    # print("u0",(u0))
    # print("u0",type(u0))

    v0 = magP ** 2 * (a2.T.dot(a3))
    # print("v0",(v0))
    # print("v0",type(v0))

    print("u0, v0 = %f, %f\n" % (u0, v0))

    # Finding alpha v
    av = np.sqrt(magP ** 2 * (a2.T.dot(a2)) - v0 ** 2)   
    # print("av",(av))
    # print("av",type(av))


    # Finding s
    a1xa3 = np.cross(a1.T, a3.T)
    a2xa3 = np.cross(a2.T, a3.T)
    s = np.nan_to_num((magP ** 4) / av )* a1xa3.dot(a2xa3.T)
    # print("s",(s))
    # print("s",type(s))
    print("s = %f\n" % s)

    # Finding alpha u
    au = np.sqrt(magP ** 2 * (a1.T.dot(a1)) - s ** 2 - u0 ** 2)
    # print("au",(au))
    # print("au",type(au))
    
    print("alphaU,alphaV = %f, %f\n" % (au, av))

    # Finding k*
    k_star = np.array([[au, s, u0],[0, av, v0],[0, 0, 1]])
    print("K* = %s\n" % k_star)

    # Findinf the unknown sign of p
    E = np.sign(b[2])

    # Finding T*
    t_star = E * magP * np.linalg.inv(k_star).dot(b).T
    print("T* = %s\n" % t_star)
    
    # Finding r3
    r3 = E * magP * a3

    # Finding r1
    r1 = magP ** 2 / av * a2xa3

    # Finding r2
    r2 = np.cross(r3, r1)

    # finding R*
    R_star = np.array([r1.T, r2.T, r3.T])
    print("R* = %s\n" % R_star)

   
   
   
   
   
  



if __name__ == '__main__':
    main()