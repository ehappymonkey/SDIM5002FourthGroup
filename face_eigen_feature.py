import cv2
import numpy as np


if __name__ == "__main__":

    imgs_mean = np.loadtxt('mean.txt')
    eig_vector = np.fromfile('eig_vector.bin',dtype = np.float32)
    eig_vector = eig_vector.reshape(10304,-1)
    u = eig_vector[:,:256]
    for i in range(0,256):
        face = u[:,i] + imgs_mean
        face = face * 255
        face[face<0] = 0
        face[face>255]=255
        face = face.astype(np.uint8).reshape(112,-1)
        cv2.imwrite('eigface/%s.bmp'%i,face)

    for i in range(1,41):
        for j in range(1,11):
            file = "ORL/s%d/%d.bmp" % (i,j)
            img = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
            img = img.astype(np.float32)/255.0
            img = img.reshape(-1)
            img = img - imgs_mean
            prj = img.dot(u)
            np.savetxt('features/%s_%s.txt'%(i,j),prj,fmt="%.5f")
            cons = u.dot(prj)
            cons = cons + imgs_mean
            cons = cons*255
            cons[cons<0]=0
            cons[cons>255]=255
            cons = cons.astype(np.uint8).reshape(112,-1)
            cv2.imwrite('construct/%s_%s.bmp'%(i,j),cons)
