from flask import Flask, Response, request, jsonify, render_template
import numpy as np
import torch
from torch import nn
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cv2
import datetime
import albumentations as A
from albumentations.pytorch import ToTensorV2
from linformer import Linformer
from vit_pytorch.efficient import ViT
from torch.utils.data import Dataset, DataLoader
import os
import calendar;
import time;


app = Flask(__name__)


@app.route('/')
def hello():
    return render_template("index.html")


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if(request.method == "POST"):
        file = request.files['file']
        temp_name = os.path.splitext(file.filename)

        #add timestamp to filename
        filename = temp_name[0] + str(calendar.timegm(time.gmtime())) + temp_name[1]


        main_transform = A.Compose(
            [
                A.Resize(224, 224),
                ToTensorV2(),
            ])

        file.save(filename)
        test_images = video_to_image(filename)

        test_dataset = DaiseeDataset(
            test_images, augmentations=main_transform)

        batch_size = len(test_images)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False)

        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        # Line transformer
        efficient_transformer = Linformer(
            dim=128,
            seq_len=49+1,  # 7x7 patches + 1 cls-token
            depth=12,
            heads=8,
            k=64
        ).to(device)

        # Visual transformer
        model = ViT(
            dim=128,
            image_size=224,
            patch_size=32,
            num_classes=4,
            transformer=efficient_transformer,
            channels=3,
        ).to(device)

        model.load_state_dict(torch.load(
            'daisee_vit_model_weights.pth', map_location=device))

        classes = {0: 'boredom', 1: 'confusion',
                   2: 'engagement', 3: 'frustration'}

        model.eval()
        for data in test_loader:
            data = data.type(torch.float)
            predict_values = model(data.to(device))
            preds = F.softmax(predict_values, dim=1)
            # print(preds)
            preds = torch.argmax(preds, dim=1)
            print(len(preds))
            preds = preds.detach().numpy()
            final_pred = np.bincount(preds).argmax()
            print(final_pred)
            if final_pred in [0, 1, 2, 3]:
                prediction = classes[final_pred]
            else:
                prediction = "not found"

        return prediction


def video_to_image(video):
    frames = []
    vidcap = cv2.VideoCapture(video)

    def getFrame(sec):
        vidcap.set(cv2.CAP_PROP_FPS, 1)
        hasFrames, image = vidcap.read()
        if hasFrames:
            # cv2.imwrite("image"+str(count)+".jpg", image)
            frames.append(image)     # save frame as JPG file
        return hasFrames
    sec = 0
    frameRate = 0.5  # //it will capture image in each 0.5 second
    count = 1
    success = getFrame(sec)
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(sec)
    return frames


class DaiseeDataset(torch.utils.data.Dataset):
    def __init__(self, img_list, augmentations):
        super(DaiseeDataset, self).__init__()
        self.img_list = img_list
        self.augmentations = augmentations

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = self.img_list[idx]
        return self.augmentations(image=img)['image']


if __name__ == '__main__':
    app.run()
