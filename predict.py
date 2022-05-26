import torch
from PIL import Image
from yolo import YOLO
import os
from pdf2image import convert_from_path
from langdetc import LangNet,DetectLan
from NER import NER
from utils import resize
if __name__ == "__main__":
    yolo = YOLO()
    dir_origin_path = "gg/"
    file1 = open("temp5.txt",'w',encoding="UTF-8")



    for i in os.listdir(dir_origin_path):
        img =dir_origin_path+ i
        print(img[-3:])
        if img[-3:] =="pdf":

            file2 = ""
            pages = convert_from_path(img, 200, poppler_path=r'C:\Program Files\poppler-0.68.0\bin')
            lang=""
            for page in pages:
                resize(page, 1500)
                lang =DetectLan('resized_image1.jpg')
                r_image = yolo.detect_image("resized_image1.jpg", lang)
                file3 = open("temp1.txt", 'r', encoding="UTF-8")
                for line in file3:
                    file2 +=(line+" ")
                file3.close()
            if lang == "fra":
                file2 = file2.replace("\r", "")
                file2 = file2.replace("\n", "")
                file1.write(file2+"\n")
            file2=""
        else:
            image = Image.open(img)
            resize(image,1500)
            lang = DetectLan('resized_image1.jpg')
            r_image = yolo.detect_image(img,lang)
            if lang == "fra":
                file3 = open("temp1.txt", 'r', encoding="UTF-8")
                for line in file3:
                    file1.write(line + "\n")
                file3.close()