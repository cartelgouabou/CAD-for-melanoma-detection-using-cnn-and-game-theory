#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-18
# demo1 -i samples/cat_dog.png -a efficientnet_b5 -t features
# demo2 -i samples/cat_dog.png -a efficientnet_b5 -t features

from __future__ import print_function

import os.path as osp

import cv2
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import models, transforms
import os 
from grad_cam import (
    BackPropagation,
    Deconvnet,
    GradCAM,
    GuidedBackPropagation,
    occlusion_sensitivity,
)

root_path=os.getcwd()
root_path=root_path[:-9]
# if a model includes LSTM, such as in image captioning,
# torch.backends.cudnn.enabled = False


def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        print("Device:", torch.cuda.get_device_name(current_device))
    else:
        print("Device: CPU")
    return device


def load_images(image_paths, input_size):
    images = []
    raw_images = []
    print("Images:")
    for i, image_path in enumerate(image_paths):
        print("\t#{}: {}".format(i, image_path))
        image, raw_image = preprocess(image_path, input_size)
        images.append(image)
        raw_images.append(raw_image)
    return images, raw_images


def get_classtable():
    classes = []
    with open("samples/synset_words.txt") as lines:
        for line in lines:
            line = line.strip().split(" ", 1)[1]
            line = line.split(", ", 1)[0].replace(" ", "_")
            classes.append(line)
    return classes


class GrayWorld(object):
    def __call__(self, img):
        img = np.asarray(img)
        result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        result = Image.fromarray(result.astype('uint8'), 'RGB')
        return result


def preprocess(image_path, input_size):
    raw_image = Image.open(image_path)
    raw_image = transforms.Resize((input_size, input_size))(raw_image)
    image = transforms.Compose(
        [
            # transforms.Resize((input_size, input_size)),
            GrayWorld(),
            transforms.ToTensor(),
            transforms.Normalize([0.59451115, 0.59429777, 0.59418184],
                                 [0.14208533, 0.18548788, 0.20363748])
        ]
    )(raw_image.copy())

    return image, raw_image


def save_gradient(filename, gradient):
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0
    cv2.imwrite(filename, np.uint8(gradient))


def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    raw_image = np.asarray(raw_image)
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(float) + raw_image.astype(float)) / 2
    cv2.imwrite(filename, np.uint8(gcam))


def save_sensitivity(filename, maps):
    maps = maps.cpu().numpy()
    scale = max(maps[maps > 0].max(), -maps[maps <= 0].min())
    maps = maps / scale * 0.5
    maps += 0.5
    maps = cm.bwr_r(maps)[..., :3]
    maps = np.uint8(maps * 255.0)
    maps = cv2.resize(maps, (224, 224), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(filename, maps)


# torchvision models
model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


def demo1(image_paths, target_layer, arch, topk, output_dir, cuda, classes, backprop=False, guided_grad=False,
          gradients_aff=False, deconvol=False):
    """
    Visualize model responses given multiple images
    """

    # name2model = {
    #     'b0': models.efficientnet_b0(),
    #     'b1': models.efficientnet_b1(),
    #     'b2': models.efficientnet_b2(),
    #     'b3': models.efficientnet_b3(),
    #     'b4': models.efficientnet_b4(),
    #     'b5': models.efficientnet_b5(),
    #     'b6': models.efficientnet_b6(),
    #     'b7': models.efficientnet_b7(),
    # }

    name2size = {
        'b0': 224,
        'b1': 240,
        'b2': 260,
        'b3': 300,
        'b4': 380,
        'b5': 456,
        'b6': 528,
        'b7': 600,
    }

    device = get_device(cuda)

    # Synset words
    # classes = get_classtable()
    # classes = ('MEL', 'NV', 'BKL')
    # classes = ('MEL', 'BKL')

    dico = {
        'bekVSmel': ('BEK','MEL'),
        'bekVSnev': ('BEK','NEV'),
        'melVSnev': ('MEL','NEV'),
        'multi':('BEK','MEL','NEV')
    }
    #le_nom = dico[classes[0]] + dico[classes[1]]
    task=classes
    store_name = '_'.join(['isic2018', 'efficientb3', task,'CE', 'W' ,'CS','F',str(100)])
    
    # Model from torchvision
    ##model = name2model["b3"]
    # input_size = name2size["b5"]
    input_size = 300
    if 'multi' in arch:
        ##model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
        topk=3
        model=torch.load(root_path+'/model/efficientNetB3_multi.pth')
    else:
        ##model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
        topk=2
        model=torch.load(root_path+'/model/efficientNetB3_bin.pth')
    # model = models.__dict__[arch](pretrained=True)
    model.load_state_dict(torch.load(
        root_path+'/checkpoint/'+store_name+'/'+'best_model_run5.pth', map_location=torch.device('cpu')))
    model.to(device)
    model.eval()

    # Images
    images, raw_images = load_images(image_paths, input_size)
    images = torch.stack(images).to(device)

    """
    Common usage:
    1. Wrap your model with visualization classes defined in grad_cam.py
    2. Run forward() with images
    3. Run backward() with a list of specific classes
    4. Run generate() to export results
    """

    # =========================================================================
    print("Vanilla Backpropagation:")

    bp = BackPropagation(model=model)
    probs, ids = bp.forward(images)  # sorted

    for i in range(topk):
        bp.backward(ids=ids[:, [i]])
        gradients = bp.generate()

        # Save results as image files
        for j in range(len(images)):
            print("\t#{}: {} ({:.5f})".format(j, dico[task][ids[j, i]], probs[j, i]))

            if gradients_aff:
                save_gradient(
                    filename=osp.join(
                        output_dir,
                        "{}-{}-vanilla-{}.png".format(j, arch, dico[task][ids[j, i]]),
                    ),
                    gradient=gradients[j],
                )

    # Remove all the hook function in the "model"
    bp.remove_hook()

    # =========================================================================
    print("Deconvolution:")

    deconv = Deconvnet(model=model)
    _ = deconv.forward(images)

    for i in range(topk):
        deconv.backward(ids=ids[:, [i]])
        gradients = deconv.generate()

        for j in range(len(images)):
            print("\t#{}: {} ({:.5f})".format(j, dico[task][ids[j, i]], probs[j, i]))

            if deconvol:
                save_gradient(
                    filename=osp.join(
                        output_dir,
                        "{}-{}-deconvnet-{}.png".format(j, arch, dico[task][ids[j, i]]),
                    ),
                    gradient=gradients[j],
                )

    deconv.remove_hook()

    # =========================================================================
    print("Grad-CAM/Guided Backpropagation/Guided Grad-CAM:")

    gcam = GradCAM(model=model)
    _ = gcam.forward(images)

    gbp = GuidedBackPropagation(model=model)
    _ = gbp.forward(images)

    for i in range(topk):
        # Guided Backpropagation
        gbp.backward(ids=ids[:, [i]])
        gradients = gbp.generate()

        # Grad-CAM
        gcam.backward(ids=ids[:, [i]])
        regions = gcam.generate(target_layer=target_layer)

        for j in range(len(images)):
            print("\t#{}: {} ({:.5f})".format(j, dico[task][ids[j, i]], probs[j, i]))

            # Guided Backpropagation
            if guided_grad:
                save_gradient(
                    filename=osp.join(
                        output_dir,
                        "{}-{}-guided-{}.png".format(j, arch, dico[task][ids[j, i]]),
                    ),
                    gradient=gradients[j],
                )

            # Grad-CAM
            image_name=image_paths[j].split('/')[-1][:-4]
            save_gradcam(
                filename=osp.join(
                    output_dir,
                    f"{image_name}_{task}_{dico[task][ids[j, i]]}.png"),
                gcam=regions[j, 0],
                raw_image=raw_images[j],
            )

            # Guided Grad-CAM
            if guided_grad:
                save_gradient(
                    filename=osp.join(
                        output_dir,
                        "{}-{}-guided_gradcam-{}-{}.png".format(
                            j, arch, target_layer, dico[task][ids[j, i]]
                        ),
                    ),
                    gradient=torch.mul(regions, gradients)[j],
                )


