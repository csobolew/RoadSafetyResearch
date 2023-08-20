import cv2
import numpy as np
import glob
import sys

def main():
    if len(sys.argv) > 1:
        directory = str(sys.argv[1])
        test = glob.iglob(f'{directory}/*')
        f = open('output.txt', 'w')
        # Read through all images in directory
        imgs1 = [(cv2.imread(i), i) for i in test]
        safe = 0
        unsafe = 0
        for i, j in enumerate(imgs1):
            # Divide by 255 to get 0-1 range pixel values
            k = j[0] / 255.0
            # Check whether image is inverted
            if np.median(k) < 0.582:
                imgs1[i] = (k, 1, j[1])
            else:
                imgs1[i] = (k, 0, j[1])
        for i in imgs1:
            if i[1] == 0:
                # If not inverted, safe if mean of area around car is less than 0.53
                if np.mean(i[0][47:62, 25:40]) < 0.53:
                    f.write(i[2] + ' Safe' + '\n')
                    safe += 1
                else:
                    f.write(i[2] + ' Unsafe' + '\n')
                    unsafe += 1
            else:
                # If inverted, safe if mean of area around car is greater than 0.53
                if np.mean(i[0][47:62, 25:40]) > 0.53:
                    f.write(i[2] + ' Safe' + '\n')
                    safe += 1
                else:
                    f.write(i[2] + ' Unsafe' + '\n')
                    unsafe += 1
        f.write('Total safe: ' + str(safe) + '\n')
        f.write('Total unsafe: ' + str(unsafe))
        f.close()
    else:
        print('No image directory provided.')


if __name__ == '__main__':
    main()