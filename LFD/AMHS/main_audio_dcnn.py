import torchvision.models as models
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import os
import cv2 as cv

class AudioDataset(Dataset):
    def __init__(self, root_dir = "./data_preprocessed", in_train = True):
        self.root_dir = root_dir
        unhealthy_csv = pd.read_csv(os.path.join(self.root_dir, "unhealthy_samples.csv"))
        unhealthy_ids = list(unhealthy_csv["id"])
        unhealthy_labels = [0] * len(unhealthy_ids)
        num = len(unhealthy_ids)
        healthy_csv = pd.read_csv(os.path.join(self.root_dir, "healthy_samples.csv"))
        healthy_ids = list(healthy_csv["id"])
        healthy_labels = [1] * len(healthy_ids)
        if in_train:
            healthy_ids = healthy_ids[ : num]
            healthy_labels = healthy_labels[ : num]

        self.ids = healthy_ids + unhealthy_ids
        self.labels = healthy_labels + unhealthy_labels

    def __getitem__(self, item):
        unique_id = self.ids[item]
        label = self.labels[item]
        data_path = os.path.join(self.root_dir + "/wav_melspec_healthy", unique_id)
        if label == 0: # unhealthy
            data_path = os.path.join(self.root_dir+"/wav_melspec_unhealthy", unique_id)
        images = []
        for file in os.listdir(data_path):
            img = cv.imread(os.path.join(data_path, file))
            img = img/ 255.0
            img = torch.from_numpy(cv.resize(img, (224, 224)).reshape((3,224,224))).type(torch.float32)
            images.append(img)
        sample = {
            "img": images,
            "label": label
        }
        return sample

    def __len__(self):
        return len(self.ids)

class AudioVGGNet(nn.Module):
    def __init__(self):
        super(AudioVGGNet, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.features = vgg16.features
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        feature = []
        for item in x:
            f = self.features(item)
            f = self.avgpool(f)
            f = torch.flatten(f, 1)
            feature.append(f)
        feature = torch.mean(torch.stack(feature), dim = 0)
        out = self.classifier(feature)
        return out

class AudioVGGAttNet(nn.Module):
    def __init__(self):
        super(AudioVGGAttNet, self).__init__()
        print("start loading pretrained VGG...")
        vgg16 = models.vgg16(pretrained=True)
        print("loaded")
        self.features = vgg16.features
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        print("creating att layer...")
        self.classifier = nn.Sequential(
            nn.TransformerEncoderLayer(d_model = 512*7*7, nhead = 8, dim_feedforward= 512, batch_first= True),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )
        print("ok")

    def forward(self, x):
        feature = []
        for item in x:
            f = self.features(item)
            f = self.avgpool(f)
            f = torch.flatten(f, 1)
            feature.append(f)
        feature = torch.stack(feature)
        out = self.classifier(feature)
        return out

def train_AudioVGGNet():
    data = AudioDataset()
    dataloader = DataLoader(data, batch_size=64, shuffle=True)

    model = AudioVGGNet()
    for param in model.features.parameters():
        param.requires_grad = False
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
    loss_func = torch.nn.CrossEntropyLoss()

    gpus = [0]
    cuda_gpu = torch.cuda.is_available()
    if (cuda_gpu):
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    total_epoch = 100
    for epoch in range(total_epoch):
        for step, batch_data in enumerate(dataloader):
            x = batch_data["img"]
            y = batch_data["label"]
            if (cuda_gpu):
                x = x.cuda()
                y = y.cuda()
            out = model(x)
            loss = loss_func(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prediction = []
            if (cuda_gpu):
                out = out.cpu().numpy()
            for item in out:
                if item[0] > item[1]:
                    prediction.append(0)
                else:
                    prediction.append(1)
            pred_y = np.array(prediction)
            target_y = batch_data["label"].data.numpy()
            print(
                "epoch %d, step %d, train acc %.4f, loss %.4f" % (epoch, step, accuracy_score(pred_y, target_y), loss))
    torch.save(model, "audionet_vgg.pth")

if __name__ == "__main__":
    print("point 1, start")
    model = AudioVGGAttNet()
    print(model)