import cv2
import numpy as np


if __name__ == "__main__":
    feats = np.zeros([400,256])
    for i in range(0,40):
        for j in range(0,10):
            feats[i*10+j,:] = np.loadtxt('features/%d_%d.txt' % (i+1,j+1))

    means = []
    for i in range(0,40):
        x = feats[10*i:10*i+9,:]
        mean = x.sum(axis = 0)/9
        means.append(mean)

    cnt = 0
    for c in range(0,40):
        x_c = feats[10*c+9,:]
        min_score = 100000
        y_c = -1
        for i in range(0,40):
            diff = x_c - means[i]
            dis = np.sqrt(diff.dot(diff))
            if dis < min_score:
                min_score = dis
                y_c = i
        if min_score > thred:
            print('unknow')
            continue
        print("class %d is predict %d" % (c,y_c))
        if c == y_c:
            cnt += 1
    print("precision : %.3f" % (cnt/40.0))

