import colorsys
import os
import time
from langdetc import get_text
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

from yolotest import YoloBody
from utils import (cvtColor, get_anchors, get_classes, preprocess_input,
                   resize_image)
from Bboxes import DecodeBox


class YOLO(object):
    def __init__(self, **kwargs):
        self.model_path = "Data/ep100-loss0.626-val_loss.pth"
        self.anchors_mask = [[3, 4, 5], [1, 2, 3]]
        self.input_shape = [416, 416]
        self.confidence = 0.5
        self.nms_iou = 0.2
        self.letterbox_image = False
        self.cuda = False
        self.fileENG = open("tempENG2.txt", 'w', encoding='utf-8')
        self.fileFRA = open("tempFRA2.txt", 'w', encoding='utf-8')
        self.fileRUS = open("tempRUS2.txt", 'w', encoding='utf-8')
        self.class_names, self.num_classes = get_classes("Data/voc_classes.txt")
        self.anchors, self.num_anchors = get_anchors("Data/yolo_anchors.txt")

        self.bbox_util = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]),
                                   self.anchors_mask)

        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

    def generate(self):

        self.net = YoloBody(self.anchors_mask, self.num_classes)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        print('{} model, anchors, and classes loaded.'.format(self.model_path))

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    def detect_image(self, image_path, Lang):
        file = open("temp1.txt", 'w', encoding='utf-8')
        image = Image.open(image_path).convert("RGB")
        crop = Image.open(image_path).convert("RGB")

        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)

            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape,
                                                         image_shape, self.letterbox_image, conf_thres=self.confidence,
                                                         nms_thres=self.nms_iou)

            if results[0] is None:
                return image

            top_label = np.array(results[0][:, 6], dtype='int32')
            top_conf = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]

        font = ImageFont.truetype(font='Data/simhei.ttf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]
            top, left, bottom, right = box
            get_text(Lang, crop, top, left, bottom, right, file)

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        file.close()
        file = open("temp1.txt", 'r',encoding='UTF-8')
        dicit = ''
        for line in file:
            dicit += ' '
            dicit += line
        file.close()
        file = open("temp1.txt", 'w', encoding='UTF-8')
        dicit = dicit.replace("\r", "")
        dicit = dicit.replace("\n", "")
        file.write(dicit + '\n')
        if Lang=='eng':
            self.fileENG.write(dicit+image_path+'\n')
        elif Lang=='fra':
            self.fileFRA.write(dicit+image_path+'\n')
        else:
            self.fileRUS.write(dicit+image_path+'\n')
        print(dicit)

        return image
