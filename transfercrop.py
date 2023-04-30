import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
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
    

def main():
    input_path = 'result/'

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
        'train': TrainingDataset(device),
        'val': TestingDataset(device)
    }

    dset_loaders = {
        'train': DataLoader(dsets['train'], batch_size=25, shuffle=True, num_workers=0),
        'val': DataLoader(dsets['val'], batch_size=25, shuffle=True, num_workers=0)
    }

    # model = models.resnet50(weights=ResNet50_Weights.DEFAULT).to(device)

    # for param in model.parameters():
    #     param.requires_grad = False

    # model.fc = nn.Sequential(
    #     nn.Linear(2048, 128).to(device),
    #     nn.ReLU(inplace=True).to(device),
    #     nn.Linear(128, 2).to(device)
    # )

    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.fc.parameters(), lr=0.01)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    torch.cuda.empty_cache()

    # model.load_state_dict(torch.load('model99.pth'))
    # trained_model = train_model(model, criterion, optimizer, scheduler)
    # torch.save(trained_model.state_dict(), 'modelcropfixed.pth')

    sim()
    
def sim():
    model = models.resnet50()
    model.fc = nn.Sequential(
        nn.Linear(2048, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 2)
    )
    model.load_state_dict(torch.load('modelcropfixed.pth'))
    model.eval()
    validation_img_paths = [43, 69, 185, 438, 1481, 1697, 1977, 1984, 2366, 2484, 4, 626,
                            680, 789, 937, 2044, 2224, 2439, 3234, 4175, 4205, 515, 1085]

    validation_img = [1824, 1844, 1848, 1854, 1897, 2021, 2906, 3086, 3153, 3249, 3456,
                            3502, 3654, 3857, 3867, 4118, 4228, 4238]
    for k in {1, 2}:
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
        for idx, i in enumerate(img_list):
            data = np.asarray(i)
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
        # _, axes = plt.subplots(1, len(img_list),figsize=(30,5))

        runningTotal = 0
        runningCorrect = 0
        predictions = []
        for i in predic_probs:
            if i[0] > i[1]:
                predictions.append(0)
            else:
                predictions.append(1)
        predictions = np.asarray(predictions)
        for i, img in enumerate(img_list):
            runningTotal += 1
            if predictions[i] == validation_labels[i]:
                runningCorrect += 1
        print("Batch: " + str(k))
        print("Accuracy: " + str((runningCorrect/runningTotal) * 100))
        f1_scorecalc = f1_score(validation_labels, predictions)
        print("F1 Score: " + str(f1_scorecalc))
        del(pred_tensor)
    
def train_model(model, criterion, optimizer, scheduler, epochs=6):
    for epoch in range(epochs):
        print('Epoch {}/{}: '.format(epoch, epochs - 1))
        print ('LR: ', scheduler.get_last_lr())
        for state in ['train', 'val']:
            if state == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dset_loaders[state]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(state == 'train'):
                    outputs = model(inputs.to(device))
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels.squeeze())

                    if state == 'train':
                        loss.backward()
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
            epoch_loss = running_loss / len(dsets[state])
            epoch_acc = running_corrects / len(dsets[state])
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(state, epoch_loss, epoch_acc))

        print()
    return model

if __name__ == "__main__":
    main()