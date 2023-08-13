from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob
import random

def main():
    directory = 'save'
    test = glob.iglob(f'{directory}/*')
    imgs1 = [cv2.imread(i) for i in test]
    imgs = []
    for i in range(0, 10):
        imgs.append(random.choice(imgs1))
    for i, j in enumerate(imgs):
        k = j / 255.0
        # print(np.median(k))
        # print('mean' + str(np.mean(k)))
        if np.median(k) < 0.582:
            imgs[i] = (k, 1)
        else:
            imgs[i] = (k, 0)
    for i in imgs:
        if i[1] == 0:
            print('uninverted')
            if np.mean(i[0][47:62, 25:40]) < 0.53:
                print('safe')
            else:
                print('unsafe')
        else:
            print('inverted')
            if np.mean(i[0][47:62, 25:40]) > 0.53:
                print('safe')
            else:
                print('unsafe')
        print('')
    f, axarr = plt.subplots(2, 5)
    axarr[0, 0].imshow(imgs[0][0][47:62, 25:40], cmap='gray', vmin=0, vmax=1)
    axarr[0, 1].imshow(imgs[1][0][47:62, 25:40], cmap='gray', vmin=0, vmax=1)
    axarr[0, 2].imshow(imgs[2][0][47:62, 25:40], cmap='gray', vmin=0, vmax=1)
    axarr[0, 3].imshow(imgs[3][0][47:62, 25:40], cmap='gray', vmin=0, vmax=1)
    axarr[0, 4].imshow(imgs[4][0][47:62, 25:40], cmap='gray', vmin=0, vmax=1)
    axarr[1, 0].imshow(imgs[5][0][47:62, 25:40], cmap='gray', vmin=0, vmax=1)
    axarr[1, 1].imshow(imgs[6][0][47:62, 25:40], cmap='gray', vmin=0, vmax=1)
    axarr[1, 2].imshow(imgs[7][0][47:62, 25:40], cmap='gray', vmin=0, vmax=1)
    axarr[1, 3].imshow(imgs[8][0][47:62, 25:40], cmap='gray', vmin=0, vmax=1)
    axarr[1, 4].imshow(imgs[9][0][47:62, 25:40], cmap='gray', vmin=0, vmax=1)
    plt.pause(1)


if __name__ == '__main__':
    main()