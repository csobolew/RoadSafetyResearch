import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from itertools import zip_longest
import random

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def main():
    #directory = 'cart'
    # test = glob.iglob(f'{directory}/*')

    testcount = 0
    imgdirs = []
    #for i in test:
   #     imgdirs.append(i)
    imgs2 = []
    #for i in range(0, 1001):
    #    j = random.choice(imgdirs)
    #    imgs2.append(cv2.imread(j))
    im = np.load('con2_test.npz')
    te = im['recon'].tolist()
    imgs = []
    for i in te:
        for j in i:
            imgs.append(j[0])
    #imgs = []
    random.shuffle(imgs)
    # for i in imgs2:
    #     k = []
    #     k.append(i[2:66,2:66])
    #     k.append(i[68:132, 68:132])
    #     k.append(i[134:198, 134:198])
    #     k.append(i[200:264, 200:264])
    #     k.append(i[266:330, 266:330])
    #     k.append(i[332:396, 332:396])
    #     k.append(i[398:462, 398:462])
    #     k.append(i[464:528, 464:528])
    #     k.append(i[530:594, 464:528])
    #     k.append(i[596:660, 464:528])
    #     k.append(i[662:726, 464:528])
    #     k.append(i[728:792, 464:528])
    #     k.append(i[794:858, 464:528])
    #     k.append(i[860:924, 464:528])
    #     k.append(i[926:990, 464:528])
    #     k.append(i[992:1056, 464:528])
    #     k.append(i[1058:1122, 464:528])
    #     k.append(i[1124:1188, 464:528])
    #     k.append(i[1190:1254, 464:528])
    #     k.append(i[1256:1320, 464:528])
    #     imgs.append(random.choice(k))
    counter = 26
    for i in grouper(imgs, 10):
        imgs = list(i)
        f, axarr = plt.subplots(2, 5)
        axarr[0, 0].imshow(imgs[0], cmap='gray', vmin=0, vmax=1)
        # axarr[0, 0].set_title(imgs[0][1])
        axarr[0, 1].imshow(imgs[1], cmap='gray', vmin=0, vmax=1)
        # axarr[0, 1].set_title(imgs[1][1])
        axarr[0, 2].imshow(imgs[2], cmap='gray', vmin=0, vmax=1)
        # axarr[0, 2].set_title(imgs[2][1])
        axarr[0, 3].imshow(imgs[3], cmap='gray', vmin=0, vmax=1)
        # axarr[0, 3].set_title(imgs[3][1])
        axarr[0, 4].imshow(imgs[4], cmap='gray', vmin=0, vmax=1)
        # axarr[0, 4].set_title(imgs[4][1])
        axarr[1, 0].imshow(imgs[5], cmap='gray', vmin=0, vmax=1)
        # axarr[1, 0].set_title(imgs[5][1])
        axarr[1, 1].imshow(imgs[6], cmap='gray', vmin=0, vmax=1)
        # axarr[1, 1].set_title(imgs[6][1])
        axarr[1, 2].imshow(imgs[7], cmap='gray', vmin=0, vmax=1)
        # axarr[1, 2].set_title(imgs[7][1])
        axarr[1, 3].imshow(imgs[8], cmap='gray', vmin=0, vmax=1)
        # axarr[1, 3].set_title(imgs[8][1])
        axarr[1, 4].imshow(imgs[9], cmap='gray', vmin=0, vmax=1)
        # axarr[1, 4].set_title(imgs[9][1])
        plt.pause(1)
        ina = input('input here: \n')
        ina = ina.split()
        labels = []
        usedimgs = []
        for i, k in zip(ina, imgs):
            j = int(i)
            if j == 0 or j == 1:
                labels.append(j)
                usedimgs.append(k)
        np.savez('cart/' + str(counter) + '.npz', img=np.array(usedimgs), lab=np.array(labels))
        counter += 1


if __name__ == '__main__':
    main()