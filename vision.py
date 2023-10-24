from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob
import random

def main():
    # directory = 'save'
    # test = glob.iglob(f'{directory}/*')
    # imgs1 = [cv2.imread(i) for i in test]
    # imgs = []
    # labels = []
    # for j in range(0, 150):
    #     labfile = np.load('labels2/' + str(j) + '.npy').tolist()
    #     for k in labfile:
    #         labels.append(k)
    # for i in imgs1:
    #     imgs.append(i)
    imgs2 = []
    labels = []
    confs = []
    for j in range(0, 101):
        labfile = np.load('labels4/' + str(j) + '.npz')
        for i in labfile['img']:
            imgs2.append(i)
            confs.append(1)
        for i in labfile['lab']:
            labels.append(i)
    imgs = []
    for i, j in enumerate(imgs2):
        if confs[i] == 1:
            k = j / 255.0
            # print(np.median(k))
            # print('mean' + str(np.mean(k)))
            if np.median(k) < 0.582:
                imgs.append((k, 1, labels[i]))
            else:
                imgs.append((k, 0, labels[i]))
    running_correct = 0
    running_total = 0
    false_positives = 0
    true_positives = 0
    total_negatives = 0
    total_positives = 0
    total_reg_positives = 0
    for l, i in enumerate(imgs):
        if l < 1500:
            if i[2] == 0:
                total_negatives += 1
            elif i[2] == 1:
                total_positives += 1
            if i[1] == 0:
                print('uninverted')
                if np.mean(i[0][47:62, 25:40]) < 0.53:
                    print('safe')
                    if i[2] == 1:
                        running_correct += 1
                        true_positives += 1
                        total_reg_positives += 1
                else:
                    print('unsafe')
                    if i[2] == 0:
                        running_correct += 1
                    elif i[2] == 1:
                        false_positives += 1
            else:
                print('inverted')
                if np.mean(i[0][47:62, 25:40]) > 0.53:
                    print('safe')
                    if i[2] == 1:
                        running_correct += 1
                        true_positives += 1
                        total_reg_positives += 1
                else:
                    print('unsafe')
                    if i[2] == 0:
                        running_correct += 1
                    elif i[2] == 1:
                        false_positives += 1
            running_total += 1
            print('')
    print('accuracy: ' + str(running_correct / running_total))
    print('false positive rate: ' + str(false_positives / total_negatives))
    recall = true_positives/total_positives
    precision = true_positives/total_reg_positives
    print('f1: ' + str(2*((precision*recall)/(precision+recall))))
    # f, axarr = plt.subplots(2, 5)
    # axarr[0, 0].imshow(imgs[0][0][47:62, 25:40], cmap='gray', vmin=0, vmax=1)
    # axarr[0, 1].imshow(imgs[1][0][47:62, 25:40], cmap='gray', vmin=0, vmax=1)
    # axarr[0, 2].imshow(imgs[2][0][47:62, 25:40], cmap='gray', vmin=0, vmax=1)
    # axarr[0, 3].imshow(imgs[3][0][47:62, 25:40], cmap='gray', vmin=0, vmax=1)
    # axarr[0, 4].imshow(imgs[4][0][47:62, 25:40], cmap='gray', vmin=0, vmax=1)
    # axarr[1, 0].imshow(imgs[5][0][47:62, 25:40], cmap='gray', vmin=0, vmax=1)
    # axarr[1, 1].imshow(imgs[6][0][47:62, 25:40], cmap='gray', vmin=0, vmax=1)
    # axarr[1, 2].imshow(imgs[7][0][47:62, 25:40], cmap='gray', vmin=0, vmax=1)
    # axarr[1, 3].imshow(imgs[8][0][47:62, 25:40], cmap='gray', vmin=0, vmax=1)
    # axarr[1, 4].imshow(imgs[9][0][47:62, 25:40], cmap='gray', vmin=0, vmax=1)
    # plt.pause(1)


if __name__ == '__main__':
    main()