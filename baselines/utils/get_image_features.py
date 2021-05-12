import json
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from torchvision import transforms
import os
from PIL import Image


resnet101 = models.resnet101(pretrained=True)

modules = list(resnet101.children())[:-1]

resnet101 = nn.Sequential(*modules)
resnet101.eval() # no fine-tuning

test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def vectorize_images(pretrained_model, img_folder):
    """
    this function extracts visual features from images
    images need to be downloaded by using urls in ../data/IMG_urls.txt
    images need to be saved in ~/images/
    """
    features = dict()  # img_id -> features

    count = 0

    for file in os.listdir(img_folder):
        # print(file)
        if file.endswith('jpg'): # in str(file):
            # print('found image')
            file_path = os.path.join(img_folder, file)
            img_id = file.split('.')[0]  # .split('_')[1] # COCO_train2014_000038920

            count += 1
            print(count, img_id)

            image = Image.open(file_path).convert('RGB')

            image = test_transform(image)
            # print(image.shape) #(3,224,224)

            input = Variable(image).view(1, image.shape[0], image.shape[1], image.shape[2])
            output = pretrained_model(input)
            output = output[0].squeeze(1).squeeze(1).data
            # print(output.shape) #torch.Size([2048])

            features[img_id] = output.numpy().tolist()

    print(count)

    # create directory where to save image features
    if not os.path.exists('../resnet101/'):
        os.makedirs('../resnet101/')
    file_name = '../resnet101/features_resnet101.json'

    with open(file_name, 'w') as file:
        json.dump(features, file)

"""
extract features
"""
images_path = '../../images'
vectorize_images(pretrained_model=resnet101, img_folder=images_path)