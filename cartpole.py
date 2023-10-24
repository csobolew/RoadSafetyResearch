import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
from math import atan2, degrees
from sklearn.linear_model import LinearRegression

safe = True
directory = 'cart'
test = glob.iglob(f'{directory}/*')
imgs = []
labs = []
correct = 0
total = 0
false_pos = 0
true_neg = 0
true_pos = 0
false_neg = 0
for i in test:
    temp = np.load(i)
    im = temp['img']
    lab = temp['lab']
    for j in zip(im, lab):
        imgs.append(np.stack((((j[0] * 255).astype(np.uint8)),)*3, axis=-1)[0:33, :])
        labs.append(j[1])
count = 0
for l in zip(imgs, labs):
    t = 175
    test2 = l[0] < t
    test2 = test2.astype(np.uint8) * 255
    count_pix = 0
    for i in test2:
        for j in i:
            if j[0] > 0:
                count_pix += 1
    if count_pix < 30:
        safe = False
    angle = 0
    if safe:
        xlist = []
        ylist = []
        for i, y in enumerate(test2):
            for j, x in enumerate(y):
                if x[0] == 255:
                    xlist.append(j)
                    ylist.append(i)
        xlist = np.array(xlist).reshape(-1, 1)
        ylist = np.array(ylist).reshape(-1, 1)
        line = LinearRegression().fit(xlist, ylist)
        invline = LinearRegression().fit(ylist, xlist)
        stdX = np.std(xlist)
        stdY = np.std(ylist)
        vertical_skew = stdY/stdX
        if vertical_skew > 1:  # line is vertical
            y_pred = invline.predict(np.array(range(0, 64)).reshape(-1,1))
            if len(y_pred) > 1:
               if y_pred[1]-y_pred[0] != 0:
                   y_slope = 1 / (y_pred[1]-y_pred[0])
               else:
                   y_slope = 1/(y_pred[1]+0.001-y_pred[0])
            angle = degrees(atan2(y_slope, 1))
        else:
            y_pred = line.predict(np.array(range(0, 64)).reshape(-1,1))
            if len(y_pred) > 1:
                y_slope = y_pred[0] - y_pred[1]
            angle = degrees(atan2(y_slope, 1))
        if abs(angle) < 40:
            safe = False
    if safe:
        total = total + 1
        if l[1] == 1:
            print('True Positive')
            true_pos = true_pos + 1
            correct = correct + 1
        else:
            print('False Positive')
            false_pos = false_pos + 1
    else:
        total = total + 1
        if l[1] == 0:
            print('True Negative')
            true_neg = true_neg + 1
            correct = correct + 1
        else:
            print('False Negative')
            false_neg = false_neg + 1
    safe = True
    count = count + 1
print('Accuracy: ' + str(correct/total))
print('False Positive Rate: ' + str(false_pos/(false_pos+true_neg)))
precision = true_pos / (true_pos + false_pos)
recall = true_pos / (true_pos + false_neg)
f1 = 2*((precision*recall)/(precision+recall))
print('Positives:', str(true_pos+false_pos))
print('Negatives:', str(true_neg+false_neg))
print('F1 score: ' + str(f1))