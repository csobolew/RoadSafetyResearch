import numpy as np
from PIL import Image
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F

def main():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
    model = models.resnet50()
    model.fc = nn.Sequential(
        nn.Linear(2048, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 2)
    )
    model.load_state_dict(torch.load('modelcropfixed.pth', map_location=torch.device('cpu')))
    model.eval()
    batch_size = 64
    imglist = torch.rand((batch_size, 1, 64, 64))
    imgs = [k for k in imglist.numpy()]
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
    predictions = []
    for i in predic_list:
        if i[0] > i[1]:
            predictions.append(0)
        else:
            predictions.append(1)
    for i in predictions:
        print(i)


if __name__ == '__main__':
    main()