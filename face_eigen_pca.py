import cv2
import numpy as np


if __name__ == "__main__":
    img_list = []
    for i in range(1,41):
        for j in range(1,11):
            file = "ORL/s%d/%d.bmp" % (i,j)
            img = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
            img = img.astype(np.float32)/255.0
            img_list.append(img)

    imgs = np.zeros([400,10304],dtype=np.float32)
    for i in range(0,400):
        imgs[i,:] = img_list[i].reshape(-1)

    imgs_mean = imgs.sum(axis=0)/400.0
    np.savetxt('mean.txt',imgs_mean,fmt="%.5f")
    imgs = imgs - imgs_mean
    #conv = imgs.transpose(1,0).dot(imgs)
    conv = imgs.dot(imgs.transpose(1,0))
    eig_value,eig_vector = np.linalg.eig(conv)
    eig_value = eig_value.astype(np.float32)
    eig_vector = eig_vector.astype(np.float32)
    eig_vector = imgs.transpose(1,0).dot(eig_vector)
    np.savetxt('eig_vetor.txt',eig_vector,fmt="%.5f")
    np.savetxt('eig_value.txt',eig_value,fmt="%.5f")
    eig_vector.tofile('eig_vector.bin')
