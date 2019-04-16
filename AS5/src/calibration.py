import cv2
import numpy as np
import sys

def main():
    file_name = sys.argv[1]
    object_point, image_point = [], []
    f = open(file_name)
    file_data = f.readlines()
    for line in file_data:
        points = line.split()
        object_point.append([float(p) for p in points[:3]])
        image_point.append([float(p) for p in points[3:]])

    A = []
    zero = np.zeros(4)
    for i, j in zip(object_point, image_point):
        pi = np.array(i)

        # Point in 2DH to 3DH concatinate with 1 
        pi = np.concatenate([pi, [1]])
        
        # print("j0",j[0])
        # print("j1",j[1])
        xipi = j[0] * pi
        yipi = j[1] * pi
        A.append(np.concatenate([pi, zero, -xipi]))
        A.append(np.concatenate([zero, pi, -yipi]))  
    
    M = []
    u, d, v = np.linalg.svd(A, full_matrices = True)
    # print("v",v)
    
    # Column of v belonging to zero singular value
    # Reshape and transpose can be used interchangably
    M = v[-1].reshape(3, 4)
    # print("M", M)

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
   
      
    


    # Calculating the Mean Square Error
    m1 = M[0][:4]
    m2 = M[1][:4]
    m3 = M[2][:4]
    mse = 0
    for i, j in zip(object_point, image_point):
        xi = j[0]
        yi = j[1]
        p = np.array(i)
        p = np.concatenate([p, [1]])
        image_xi = (m1.T.dot(p)) / (m3.T.dot(p))
        image_yi = (m2.T.dot(p)) / (m3.T.dot(p))
        mse += ((xi - image_xi) ** 2 + (yi - image_yi) ** 2)
    mse = mse / len(object_point)
    
    print("\nMean Square Error = %s\n" % mse)


if __name__ == '__main__':
    main()