# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import shutil

import math
import os
import cv2
import sys
import argparse
from pathlib import Path
from typing import Iterable
from PIL import Image, ImageDraw, ImageFont
import numpy as np

import torch

import util.misc as utils

from models import build_model
from datasets.drone import make_Drone_transforms


import matplotlib.pyplot as plt
import time

import datasets.transforms as T

# Postprocess the predictions to fetch the image and thje bounding box around objects

# standard pyTorch mean-std input image normalization
# colors for visualisation purposes 

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], 
           [0.929, 0.694, 0.125], [0.494, 0.184, 0.556],
           [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

CLASSES = ['', 'Drone']

'''
No need for further transformations as the model is already trained on augmented data
'''
# transform = T.compose([
#     T.Resize(800),
#     T.ToTensor(),
#     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])


def box_cxcywh_to_xyxy(x):
    ''' for output bounding box post-processing
    '''
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    ''' for output bounding box post-processing
    '''
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h,
                          img_w, img_h
                          ], dtype=torch.float32)
    return b

def get_images(in_path):
    img_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))

    return img_files

def detect(image, model, transform):
    for img in image:
        filename = os.path.basename(img)
        print("detecting... ")
        im = transform(filename).unsqueeze(0)
        assert im.shape[-2] <= 416 and im.shape[-1] <= 1600
        outputs = model(im)
        # keep only predictions with 0.7+ confidence
        probas = outputs['pred_logits'].softmax(-1)[0,:,:-1]
        keep = probas.max(-1).values > 0.7
        # convert boxes from [0;1] to image scales
        bbox_scaled = rescale_bboxes(['pred_boxes'][0, keep], im.size)
        return probas[keep], bbox_scaled

def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=c, linewidth=3))
        
        cl = p.argmax()
        text = f'{CLASSES[cl]: {p[cl]:.2f}}'
        ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=6, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=10, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default="drone")
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--data_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save the results, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    parser.add_argument('--thresh', default=0.5, type=float)

    return parser


@torch.no_grad()
def infer(images_path, model, postprocessors, device, output_path):
    model.eval()
    duration = 0
    
    for img_sample in images_path:
        
        filename = os.path.basename(img_sample)
        print("processing...{}".format(filename))
        im = cv2.imread(img_sample)
        orig_image = Image.open(img_sample)
        w, h = orig_image.size
        transform = make_Drone_transforms("val")

        dummy_target = {
            "size": torch.as_tensor([int(h), int(w)]),
            "orig_size": torch.as_tensor([int(h), int(w)])
        }

        image, targets = transform(orig_image, dummy_target)

        image = image.unsqueeze(0)
        assert im.shape[-2] <= 416 and im.shape[-1] <= 416
        
        image = image.to(device)


        start_t = time.perf_counter()
        outputs = model(image)
        end_t = time.perf_counter()
        
        probas = outputs['pred_logits'].softmax(-1)[0,:,:-1]
        keep = probas.max(-1).values > 0.7
        print('keep is:\n', keep)
        # convert boxes from [0;1] to image scales
        # bbox_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], image.size)
       
        print(f"scores over 0.7 thresholds are: {probas[keep]}")
        # print(f"boxes are {bbox_scaled}")
        # as before...
        outputs["pred_logits"] = outputs["pred_logits"].cpu()
        outputs["pred_boxes"] = outputs["pred_boxes"].cpu()
        
        pred_logits=outputs['pred_logits'][0][:, :]
        pred_boxes=outputs['pred_boxes'][0]
        # print(pred_logits)
        max_output = pred_logits.softmax(-1).max(-1)
        topk = max_output.values.topk(1)
        pred_logits = pred_logits[topk.indices]
        pred_boxes = pred_boxes[topk.indices]
        # bboxes_scaled = rescale_bboxes(pred_boxes, orig_image.size)
        # probas = probas[keep].cpu().data.numpy()
        x, y, w, h = pred_boxes.numpy()[0]
        
        
        logit = pred_logits.numpy()[0][-1]
        OR = np.exp(logit)
        conf = OR/(1+OR)

        infer_time = end_t - start_t
        duration += infer_time
        print("Processing...{} ({:.3f}s)".format(filename, infer_time))
        

        im = cv2.rectangle(im,(int((x-w/2)*416),int((y-h/2)*416)),(int((x+w/2)*416),int((y+h/2)*416)),(0,0,255),2)
        cv2.putText(im,'Drone: '+str(round(conf,4)),(int((x-w/2)*416),int((y-h/2)*416)-10),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),2)
        filesave = 'Detect/exp/'+filename
        print("#########")
        cv2.imwrite(filesave,im)


        
    avg_duration = duration / (len(images_path))
    print("Avg. Time: {:.3f}s".format(avg_duration))


if __name__ == "__main__":
    # load the trained model
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    # load the trained model 
    model, _, postprocessors = build_model(args)

    # resume from checkpoint! 
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    
    model.to(device)
    
    # create folder for collecting output results 
    shutil.rmtree('/home/cattivediferoce/Desktop/detr/detr/Detect/exp')
    os.mkdir('/home/cattivediferoce/Desktop/detr/detr/Detect/exp')

    # retrieve from args the image data path 
    image_paths = get_images(args.data_path)
    
    # for _ in range(2):
    #     image = image_paths[0]
    #     scores, boxes = detect(image, model, transform)
    #     plot_results(image, scores, boxes)



    infer(image_paths, model, postprocessors, device, args.output_dir)
