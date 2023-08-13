from itertools import zip_longest

import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from torch.utils.data.sampler import WeightedRandomSampler
import torch.nn as nn
from torchvision.models import ResNet50_Weights
from torch.nn import functional as F
import torch.optim as optim
import re
import glob
import cv2
from sklearn.metrics import f1_score

class TrainingDataset(Dataset):
    def __init__(self, device):
        self.transform = dtransforms['train']
        self.imgs_path = "result/"
        self.device = device
        self.img_path = "20grey_test_series.npz"
        test = np.load(self.img_path)['series']
        test1 = np.load(self.img_path)['safe']
        self.data = []
        file_list = glob.glob(self.imgs_path + "*")
        self.class_map = {'unsafe': 0, 'safe': 1}
        for class_path in file_list:
            classes_name = class_path.split("/")[-1]
            class_name = classes_name.split(".")[0]
            num = int(re.split(r'\D+',class_name)[0])
            type = re.split(str(num), class_name)[-1]
            img_path = class_path.split("/")[0] + '/' + class_name + '.png'
            if (num % 2) == 0:
                if type == 'ori':
                    for imagenum in range(10):
                        if (test1[num][imagenum] == 0):
                            self.data.append([img_path, imagenum, 'unsafe'])
                        else:
                            self.data.append([img_path, imagenum, 'safe'])
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        image, number, label = self.data[idx]
        img = cv2.imread(image)
        if number == 0:
            img = img[2:66, 2:66]
        elif number == 1:
            img = img[2:66, 68:132]
        elif number == 2:
            img = img[2:66, 134:198]
        elif number == 3:
            img = img[2:66, 200:264]
        elif number == 4:
            img = img[2:66, 266:330]
        elif number == 5:
            img = img[2:66, 332:396]
        elif number == 6:
            img = img[2:66, 398:462]
        elif number == 7:
            img = img[2:66, 464:528]
        elif number == 8:
            img = img[68:132, 2:66]
        elif number == 9:
            img = img[68:132, 68:132]
        class_id = self.class_map[label]
        # img_tensor = torch.from_numpy(img).to('cuda')
        # img = cv2.Canny(img, 50, 150)
        # img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        data = np.asarray(img)
        data = data[45:65, 25:45]
        min = np.min(data)
        mask = data < min+2
        coords = np.argwhere(mask)
        newCoords = list()
        for j in coords:
            newCoords.append(j)
        newCoords = np.asarray(newCoords)
        xy0 = newCoords.min(axis=0)
        x0 = xy0[0]
        y0 = xy0[1]
        xy1 = newCoords.max(axis=0) + 1
        x1 = xy1[0]
        y1 = xy1[1]
        center = (x0 + x1) / 2, (y0 + y1) / 2
        img = img[int(center[0])-10+45:int(center[0])+10+45, int(center[1])-10+25:int(center[1])+10+25]
        img = Image.fromarray(img)
        img_tensor = self.transform(img).to(self.device)
        class_id = torch.tensor([class_id]).to(self.device)
        return img_tensor, class_id
    
class TestingDataset(Dataset):
    def __init__(self, device):
        self.transform = dtransforms['val']
        self.imgs_path = "result/"
        self.device = device
        self.img_path = "20grey_test_series.npz"
        test = np.load(self.img_path)['series']
        test1 = np.load(self.img_path)['safe']
        self.data = []
        file_list = glob.glob(self.imgs_path + "*")
        self.class_map = {'unsafe': 0, 'safe': 1}
        for class_path in file_list:
            classes_name = class_path.split("/")[-1]
            class_name = classes_name.split(".")[0]
            num = int(re.split(r'\D+',class_name)[0])
            type = re.split(str(num), class_name)[-1]
            img_path = class_path.split("/")[0] + '/' + class_name + '.png'
            if (num % 2) != 0:
                if type == 'ori':
                    for imagenum in range(10):
                        if (test1[num][imagenum] == 0):
                            self.data.append([img_path, imagenum, 'unsafe'])
                        else:
                            self.data.append([img_path, imagenum, 'safe'])
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        image, number, label = self.data[idx]
        img = cv2.imread(image)
        if number == 0:
            img = img[2:66, 2:66]
        elif number == 1:
            img = img[2:66, 68:132]
        elif number == 2:
            img = img[2:66, 134:198]
        elif number == 3:
            img = img[2:66, 200:264]
        elif number == 4:
            img = img[2:66, 266:330]
        elif number == 5:
            img = img[2:66, 332:396]
        elif number == 6:
            img = img[2:66, 398:462]
        elif number == 7:
            img = img[2:66, 464:528]
        elif number == 8:
            img = img[68:132, 2:66]
        elif number == 9:
            img = img[68:132, 68:132]
        # img = cv2.Canny(img, 50, 150)
        # img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        data = np.asarray(img)
        data = data[45:65, 25:45]
        min = np.min(data)
        mask = data < min+2
        coords = np.argwhere(mask)
        newCoords = list()
        for j in coords:
            newCoords.append(j)
        newCoords = np.asarray(newCoords)
        xy0 = newCoords.min(axis=0)
        x0 = xy0[0]
        y0 = xy0[1]
        xy1 = newCoords.max(axis=0) + 1
        x1 = xy1[0]
        y1 = xy1[1]
        center = (x0 + x1) / 2, (y0 + y1) / 2
        img = img[int(center[0])-10+45:int(center[0])+10+45, int(center[1])-10+25:int(center[1])+10+25]
        img = Image.fromarray(img)
        class_id = self.class_map[label]
        # img_tensor = torch.from_numpy(img).to('cuda')
        img_tensor = self.transform(img).to(self.device)
        class_id = torch.tensor(class_id).to(self.device)
        return img_tensor, class_id


class TrainingSet2(Dataset):
    def __init__(self, device):
        self.transform = dtransforms['train']
        self.imgs_path = "result/"
        self.device = device
        self.img_path = "thr2-pred-10k.npz"
        self.npfile = np.load(self.img_path)
        self.data = []
        self.imgsorig = self.npfile['pred']
        self.class_map = {'unsafe': 0, 'safe': 1}
        self.labels = []
        self.imgs = []
        for i in self.imgsorig:
            for j in i:
                self.imgs.append(j)
        for j in range(0, 20):
            labfile = np.load('labels/'+str(j)+'.npy').tolist()
            for k in labfile:
                self.labels.append(k)
        for i, img in enumerate(self.imgs):
            if i < 2000:
                if self.labels[i] == 1:
                    self.data.append([img, 'safe'])
                else:
                    self.data.append([img, 'unsafe'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        data = img[0]
        data = data[43:63, 25:45]
        min = np.min(data)
        mask = data < min+2
        coords = np.argwhere(mask)
        newCoords = list()
        for j in coords:
            newCoords.append(j)
        newCoords = np.asarray(newCoords)
        xy0 = newCoords.min(axis=0)
        x0 = xy0[0]
        y0 = xy0[1]
        xy1 = newCoords.max(axis=0) + 1
        x1 = xy1[0]
        y1 = xy1[1]
        center = (x0 + x1) / 2, (y0 + y1) / 2
        img = img[0][int(center[0])-10+45:int(center[0])+10+45, int(center[1])-10+25:int(center[1])+10+25]
        img = Image.fromarray(img).convert("RGB")
        class_id = self.class_map[label]
        # img_tensor = torch.from_numpy(img).to('cuda')
        img_tensor = self.transform(img).to(self.device)
        class_id = torch.tensor(class_id).to(self.device)
        return img_tensor, class_id


class TestingSet2(Dataset):
    def __init__(self, device):
        self.transform = dtransforms['val']
        self.imgs_path = "result/"
        self.device = device
        self.img_path = "thr2-pred-10k.npz"
        self.npfile = np.load(self.img_path)
        self.data = []
        self.imgsorig = self.npfile['pred']
        self.class_map = {'unsafe': 0, 'safe': 1}
        self.labels = []
        self.imgs = []
        for i in self.imgsorig:
            for j in i:
                self.imgs.append(j)
        for j in range(0, 20):
            labfile = np.load('labels/'+str(j)+'.npy').tolist()
            for k in labfile:
                self.labels.append(k)
        for i, img in enumerate(self.imgs):
            if 1999 < i < 3000:
                if self.labels[i] == 1:
                    self.data.append([img, 'safe'])
                else:
                    self.data.append([img, 'unsafe'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        data = img[0]
        data = data[43:63, 25:45]
        min = np.min(data)
        mask = data < min+2
        coords = np.argwhere(mask)
        newCoords = list()
        for j in coords:
            newCoords.append(j)
        newCoords = np.asarray(newCoords)
        xy0 = newCoords.min(axis=0)
        x0 = xy0[0]
        y0 = xy0[1]
        xy1 = newCoords.max(axis=0) + 1
        x1 = xy1[0]
        y1 = xy1[1]
        center = (x0 + x1) / 2, (y0 + y1) / 2
        img = img[0][int(center[0])-10+45:int(center[0])+10+45, int(center[1])-10+25:int(center[1])+10+25]
        img = Image.fromarray(img).convert("RGB")
        class_id = self.class_map[label]
        # img_tensor = torch.from_numpy(img).to('cuda')
        img_tensor = self.transform(img).to(self.device)
        class_id = torch.tensor(class_id).to(self.device)
        return img_tensor, class_id



class TrainingSet3(Dataset):
    def __init__(self, device):
        self.transform = dtransforms['train']
        self.directory = 'save/'
        self.device = device
        self.data = []
        self.class_map = {'unsafe': 0, 'safe': 1}
        self.labels = []
        self.imgs = []
        for j in range(0, 100):
            labfile = np.load('labels2/'+str(j)+'.npy').tolist()
            for k in labfile:
                self.labels.append(k)
        for i, im in enumerate(glob.iglob(f'{self.directory}/*')):
            if i < 1000:
                if self.labels[i] == 1:
                    self.data.append([cv2.imread(im, 0) / 255.0, 'safe'])
                else:
                    self.data.append([cv2.imread(im, 0) / 255.0, 'unsafe'])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        data = img
        data = data[43:63, 25:45]
        min = np.min(data)
        mask = data < min+2
        coords = np.argwhere(mask)
        newCoords = list()
        for j in coords:
            newCoords.append(j)
        newCoords = np.asarray(newCoords)
        # xy0 = newCoords.min(axis=0)
        # x0 = xy0[0]
        # y0 = xy0[1]
        # xy1 = newCoords.max(axis=0) + 1
        # x1 = xy1[0]
        # y1 = xy1[1]
        # center = 54, 32
        # img = img[int(center[0])-10+45:int(center[0])+10+45, int(center[1])-10+25:int(center[1])+10+25]
        img = img[43:63, 25:45]
        img = Image.fromarray(img).convert("RGB")
        class_id = self.class_map[label]
        # img_tensor = torch.from_numpy(img).to('cuda')
        img_tensor = self.transform(img).to(self.device)
        class_id = torch.tensor(class_id).to(self.device)
        return img_tensor, class_id


class TestingSet3(Dataset):
    def __init__(self, device):
        self.transform = dtransforms['val']
        self.directory = 'save/'
        self.device = device
        self.data = []
        self.class_map = {'unsafe': 0, 'safe': 1}
        self.labels = []
        self.imgs = []
        for j in range(0, 150):
            labfile = np.load('labels2/'+str(j)+'.npy').tolist()
            for k in labfile:
                self.labels.append(k)
        for i, im in enumerate(glob.iglob(f'{self.directory}/*')):
            if 999 < i < 1500:
                if self.labels[i] == 1:
                    self.data.append([cv2.imread(im, 0) / 255.0, 'safe'])
                else:
                    self.data.append([cv2.imread(im, 0) / 255.0, 'unsafe'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        data = img
        data = data[43:63, 25:45]
        min = np.min(data)
        mask = data < min+2
        coords = np.argwhere(mask)
        newCoords = list()
        for j in coords:
            newCoords.append(j)
        newCoords = np.asarray(newCoords)
        xy0 = newCoords.min(axis=0)
        x0 = xy0[0]
        y0 = xy0[1]
        xy1 = newCoords.max(axis=0) + 1
        x1 = xy1[0]
        y1 = xy1[1]
        center = 54, 32
        img = img[43:63, 25:45]
        img = Image.fromarray(img).convert("RGB")
        class_id = self.class_map[label]
        # img_tensor = torch.from_numpy(img).to('cuda')
        img_tensor = self.transform(img).to(self.device)
        class_id = torch.tensor(class_id).to(self.device)
        return img_tensor, class_id

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def main():
    input_path = 'result/'

    # directory = 'save/'
    # num1 = 0
    # num2 = 0
    # for m, i in enumerate(tests):
    #     for n, x in enumerate(i):
    #         num1 += np.median(x[0])
    #         num2 += 1
    #         img1 = Image.fromarray(np.uint8(x[0] * 255), 'L')
    #         if np.median(x[0]) > 0.582:
    #             img1.save('imgs/' + str(m*len(i) + n) + '.png')
    # print('average median:', num1/num2)
    # counter = 0
    # for i in grouper(glob.iglob(f'{directory}/*'), 10):
    #     lists = []
    #     imgs = list(i)
    #     f, axarr = plt.subplots(2, 5)
    #     axarr[0, 0].imshow(cv2.imread(imgs[0]), cmap='gray', vmin=0, vmax=1)
    #     axarr[0, 1].imshow(cv2.imread(imgs[1]), cmap='gray', vmin=0, vmax=1)
    #     axarr[0, 2].imshow(cv2.imread(imgs[2]), cmap='gray', vmin=0, vmax=1)
    #     axarr[0, 3].imshow(cv2.imread(imgs[3]), cmap='gray', vmin=0, vmax=1)
    #     axarr[0, 4].imshow(cv2.imread(imgs[4]), cmap='gray', vmin=0, vmax=1)
    #     axarr[1, 0].imshow(cv2.imread(imgs[5]), cmap='gray', vmin=0, vmax=1)
    #     axarr[1, 1].imshow(cv2.imread(imgs[6]), cmap='gray', vmin=0, vmax=1)
    #     axarr[1, 2].imshow(cv2.imread(imgs[7]), cmap='gray', vmin=0, vmax=1)
    #     axarr[1, 3].imshow(cv2.imread(imgs[8]), cmap='gray', vmin=0, vmax=1)
    #     axarr[1, 4].imshow(cv2.imread(imgs[9]), cmap='gray', vmin=0, vmax=1)
    #     plt.pause(1)
    #     ina = input('input here: \n')
    #     ina = ina.split()
    #     for i in ina:
    #         j = int(i)
    #         lists.append(j)
    #     arr = np.asarray(lists)
    #     np.save('labels2/' + str(counter) + '.npy', arr)
    #     counter += 1


    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    global dtransforms
    dtransforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])
    }

    global dsets
    global dset_loaders
    dsets = {
        'train0': TrainingDataset(device),
        'val0': TestingDataset(device),
        'train1': TrainingSet2(device),
        'val1': TestingSet2(device),
        'train2': TrainingSet3(device),
        'val2': TestingSet3(device)
    }
    safecount = 0
    unsafecount = 0
    for i in dsets['train2'].labels:
        if i == 1:
            safecount += 1
        else:
            unsafecount += 1
    weights = []
    for i in dsets['train2'].labels:
        if i == 1:
            weights.append(1 / safecount)
        else:
            weights.append(1 / unsafecount)
    sample_weights = np.array(weights)
    sample_weights = torch.from_numpy(sample_weights)
    sampler = WeightedRandomSampler(sample_weights.type('torch.DoubleTensor'), len(sample_weights))
    dset_loaders = {
        'train0': DataLoader(dsets['train0'], batch_size=25, shuffle=True, num_workers=0),
        'val0': DataLoader(dsets['val0'], batch_size=25, shuffle=True, num_workers=0),
        'train1': DataLoader(dsets['train1'], batch_size=25, shuffle=True, num_workers=0),
        'val1': DataLoader(dsets['val1'], batch_size=25, shuffle=True, num_workers=0),
        'train2': DataLoader(dsets['train2'], batch_size=1, num_workers=0, sampler=sampler),
        'val2': DataLoader(dsets['val2'], batch_size=1, shuffle=True, num_workers=0)
    }

    model = models.resnet50(weights=ResNet50_Weights.DEFAULT).to(device)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(2048, 128).to(device),
        nn.ReLU(inplace=True).to(device),
        nn.Linear(128, 2).to(device)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    torch.cuda.empty_cache()

    model.load_state_dict(torch.load('modelcropfixed.pth'))
    trained_model = train_model(model, criterion, optimizer, scheduler)
    torch.save(trained_model.state_dict(), 'modelcropshifted.pth')

    #sim()
    #sim2()

def sim2():
    print('\n')
    print('New Data (w/ Inverted Images)')
    print('-----------------------------')
    model = models.resnet50()
    model.fc = nn.Sequential(
        nn.Linear(2048, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 2)
    )
    model.load_state_dict(torch.load('modelcropfixed.pth', map_location=torch.device('cpu')))
    model.eval()
    img_path = "thr2-pred-10k.npz"
    npfile = np.load(img_path)
    imgsorig = npfile['pred']
    imgs = []
    for i in imgsorig:
        for j in i:
            imgs.append(j)
    imgs = imgs[3600:3900]
    setList = []
    for idx, i in enumerate(imgs):
        data = i[0]
        if np.median(data) > 0.582:
            setList.append(0)
        else:
            setList.append(1)
        data = data[43:63, 25:45]
        min = np.min(data)
        mask = data < min + 2
        coords = np.argwhere(mask)
        xy0 = coords.min(axis=0)
        x0 = xy0[0]
        y0 = xy0[1]
        xy1 = coords.max(axis=0) + 1
        x1 = xy1[0]
        y1 = xy1[1]
        center = (x0 + x1) / 2, (y0 + y1) / 2
        imgs[idx] = Image.fromarray(i[0][int(center[0]) - 10 + 45:int(center[0]) + 10 + 45,
                                        int(center[1]) - 10 + 25:int(center[1]) + 10 + 25]).convert('RGB')

    validation_batch = torch.stack([dtransforms['val'](img) for img in imgs])
    pred_tensor = model(validation_batch)
    predic_probs = F.softmax(pred_tensor, dim=1).cpu().data.numpy()
    del model.fc
    del model
    del pred_tensor
    model1 = models.resnet50()
    model1.fc = nn.Sequential(
        nn.Linear(2048, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 2)
    )
    model1.load_state_dict(torch.load('modelcropboth3.pth', map_location=torch.device('cpu')))
    model1.eval()
    pred_tensor1 = model1(validation_batch)
    predic_probs1 = F.softmax(pred_tensor1, dim=1).cpu().data.numpy()
    del model1
    del pred_tensor1

    predic_list = []
    for n, (x, y) in enumerate(zip(predic_probs, predic_probs1)):
       if setList[n] == 0:
            predic_list.append(x)
       else:
            predic_list.append(y)
    validation_labels = []
    for j in range(0, 20):
        labfile = np.load('labels/' + str(j) + '.npy').tolist()
        for k in labfile:
            validation_labels.append(k)
    validation_labels = validation_labels[3600:3900]

    runningTotal = 0
    runningCorrect = 0
    predictions = []
    for i in predic_list:
        if i[0] > i[1]:
            print('test')
            predictions.append(0)
        else:
            predictions.append(1)
    predictions = np.asarray(predictions)
    for i, img in enumerate(imgs):
        runningTotal += 1
        if predictions[i] == validation_labels[i]:
            runningCorrect += 1
    print("Batch: " + str(k))
    print("Accuracy: " + str((runningCorrect / runningTotal) * 100))
    f1_scorecalc = f1_score(validation_labels, predictions)
    print("F1 Score: " + str(f1_scorecalc))



def sim():
    print('Old Data')
    print('--------')
    validation_img_paths = [43, 69, 185, 438, 1481, 1697, 1977, 1984, 2366, 2484, 4, 626,
                            680, 789, 937, 2044, 2224, 2439, 3234, 4175, 4205, 515, 1085]

    validation_img = [1824, 1844, 1848, 1854, 1897, 2021, 2906, 3086, 3153, 3249, 3456,
                            3502, 3654, 3857, 3867, 4118, 4228, 4238]
    for k in {1, 2}:
        model = models.resnet50()
        model.fc = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)
        )
        model.load_state_dict(torch.load('modelcropfixed.pth', map_location=torch.device('cpu')))
        model.eval()
        img_list = []
        if k == 1:
            for img_path in validation_img_paths:
                img_list.append(Image.fromarray(np.asarray(Image.open('result/' + str(img_path) + 'pre.png'))[2:66, 2:66]))
                img_list.append(Image.fromarray(np.asarray(Image.open('result/' + str(img_path) + 'pre.png'))[2:66, 68:132]))
                img_list.append(Image.fromarray(np.asarray(Image.open('result/' + str(img_path) + 'pre.png'))[2:66, 134:198]))
                img_list.append(Image.fromarray(np.asarray(Image.open('result/' + str(img_path) + 'pre.png'))[2:66, 200:264]))
                img_list.append(Image.fromarray(np.asarray(Image.open('result/' + str(img_path) + 'pre.png'))[2:66, 266:330]))
                img_list.append(Image.fromarray(np.asarray(Image.open('result/' + str(img_path) + 'pre.png'))[2:66, 332:396]))
                img_list.append(Image.fromarray(np.asarray(Image.open('result/' + str(img_path) + 'pre.png'))[2:66, 398:462]))
                img_list.append(Image.fromarray(np.asarray(Image.open('result/' + str(img_path) + 'pre.png'))[2:66, 464:528]))
                img_list.append(Image.fromarray(np.asarray(Image.open('result/' + str(img_path) + 'pre.png'))[68:132, 2:66]))
                img_list.append(Image.fromarray(np.asarray(Image.open('result/' + str(img_path) + 'pre.png'))[68:132, 68:132]))
        elif k == 2:
            for img_path in validation_img:
                img_list.append(Image.fromarray(np.asarray(Image.open('result/' + str(img_path) + 'pre.png'))[2:66, 2:66]))
                img_list.append(Image.fromarray(np.asarray(Image.open('result/' + str(img_path) + 'pre.png'))[2:66, 68:132]))
                img_list.append(Image.fromarray(np.asarray(Image.open('result/' + str(img_path) + 'pre.png'))[2:66, 134:198]))
                img_list.append(Image.fromarray(np.asarray(Image.open('result/' + str(img_path) + 'pre.png'))[2:66, 200:264]))
                img_list.append(Image.fromarray(np.asarray(Image.open('result/' + str(img_path) + 'pre.png'))[2:66, 266:330]))
                img_list.append(Image.fromarray(np.asarray(Image.open('result/' + str(img_path) + 'pre.png'))[2:66, 332:396]))
                img_list.append(Image.fromarray(np.asarray(Image.open('result/' + str(img_path) + 'pre.png'))[2:66, 398:462]))
                img_list.append(Image.fromarray(np.asarray(Image.open('result/' + str(img_path) + 'pre.png'))[2:66, 464:528]))
                img_list.append(Image.fromarray(np.asarray(Image.open('result/' + str(img_path) + 'pre.png'))[68:132, 2:66]))
                img_list.append(Image.fromarray(np.asarray(Image.open('result/' + str(img_path) + 'pre.png'))[68:132, 68:132]))
        if k == 1:
            validation_labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, #***
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, #515
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        elif k == 2:
            validation_labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, #1848
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        setList = []
        for idx, i in enumerate(img_list):
            data = np.asarray(i)
            if np.median(data) > 0.582:
                setList.append(0)
            else:
                setList.append(1)
            data = data[45:65, 25:45]
            min = np.min(data)
            mask = data < min+2
            coords = np.argwhere(mask)
            xy0 = coords.min(axis=0)
            x0 = xy0[0]
            y0 = xy0[1]
            xy1 = coords.max(axis=0) + 1
            x1 = xy1[0]
            y1 = xy1[1]
            center = (x0 + x1) / 2, (y0 + y1) / 2
            img_list[idx] = Image.fromarray(np.asarray(i)[int(center[0])-10+45:int(center[0])+10+45, int(center[1])-10+25:int(center[1])+10+25])

        validation_batch = torch.stack([dtransforms['val'](img) for img in img_list])
        pred_tensor = model(validation_batch)
        predic_probs = F.softmax(pred_tensor, dim=1).cpu().data.numpy()
        del pred_tensor
        del model

        model1 = models.resnet50()
        model1.fc = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)
        )
        model1.load_state_dict(torch.load('modelcropboth3.pth', map_location=torch.device('cpu')))
        model1.eval()
        pred_tensor1 = model1(validation_batch)
        predic_probs1 = F.softmax(pred_tensor1, dim=1).cpu().data.numpy()
        del model1
        del pred_tensor1

        predic_list = []
        for n, (x, y) in enumerate(zip(predic_probs, predic_probs1)):
            if setList[n] == 0:
                predic_list.append(x)
            else:
                predic_list.append(y)
        # _, axes = plt.subplots(1, len(img_list),figsize=(30,5))

        runningTotal = 0
        runningCorrect = 0
        predictions = []
        for i in predic_list:
            if i[0] > i[1]:
                predictions.append(0)
            else:
                predictions.append(1)
        predictions = np.asarray(predictions)
        for i, img in enumerate(img_list):
            runningTotal += 1
            if predictions[i] == validation_labels[i]:
                if predictions[i] == 0:
                    print('test2')
                runningCorrect += 1
        print("Batch: " + str(k))
        print("Accuracy: " + str((runningCorrect/runningTotal) * 100))
        f1_scorecalc = f1_score(validation_labels, predictions)
        print("F1 Score: " + str(f1_scorecalc))
    
def train_model(model, criterion, optimizer, scheduler, epochs=8):
    for epoch in range(epochs):
        set = 2
        print('Epoch {}/{}: '.format(epoch, epochs - 1))
        print ('LR: ', scheduler.get_last_lr())
        for state in ['train', 'val']:
            if state == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dset_loaders[state + str(set)]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(state == 'train'):
                    outputs = model(inputs.to(device))
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if state == 'train':
                        loss.backward()
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                print(name, param.grad.abs().sum())
                        optimizer.step()
                    out = outputs.detach().cpu().numpy()
                    lab = labels.detach().cpu().numpy()
                    for i, j in zip(out, lab):
                        if i[0] > i[1] and j == 0:
                            running_corrects += 1
                        elif i[0] < i[1] and j == 1:
                            running_corrects += 1
                    running_loss += loss.item() * inputs.size(0)
            scheduler.step()
            epoch_loss = running_loss / len(dsets[state + str(set)])
            epoch_acc = running_corrects / len(dsets[state + str(set)])
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(state, epoch_loss, epoch_acc))

        print()
    return model

if __name__ == "__main__":
    main()