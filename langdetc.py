import torch
import numpy as np
import pytesseract
from utils import im_cut
from PIL import Image

mas = ['fra', 'eng', 'rus']


def To_String(data):
    stri = []
    stri.append(data)
    return stri


def Dataset(file, lang):
    data = []
    for f in file:
        data.append([f, lang])
    return data


def Vectorize(data, dicti):
    result = []
    for i in data:
        vect = np.zeros(len(dicti))
        for j in i:
            for k in range(0, len(dicti)):
                if (j == dicti[k]):
                    vect[k] += 1
                    break
        result.append(vect)
    return result


def to_vec(data):
    result = []
    trigram = []
    e = 0
    for i in data:
        e += 1
        trigram.clear()
        for j in range(len(i[0]) - 2):
            trigram.append(i[0][j:j + 3])
        result.append(trigram[:])
    return result


def Take_exp(data):
    f = open("Data/dataset.txt", encoding='UTF8')
    dicit = [line.strip() for line in f]
    f.close()
    data = To_String(data)
    # print(data)
    d = Dataset(data, "val")
    dd = to_vec(d)
    ddd = Vectorize(dd, dicit)
    Valid = torch.FloatTensor(ddd)
    return Valid


class LangNet(torch.nn.Module):
    def __init__(self, n_hidden_neurons):
        super(LangNet, self).__init__()

        self.fc1 = torch.nn.Linear(513, n_hidden_neurons)
        self.activ1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(n_hidden_neurons, n_hidden_neurons)
        self.activ2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(n_hidden_neurons, n_hidden_neurons // 2)
        self.activ3 = torch.nn.ReLU()
        self.fc4 = torch.nn.Linear(n_hidden_neurons // 2, 3)
        self.sm = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activ1(x)
        x = self.fc2(x)
        x = self.activ2(x)
        x = self.fc3(x)
        x = self.activ3(x)
        x = self.fc4(x)
        return x

    def inference(self, x):
        x = self.forward(x)
        x = x.argmax(dim=1)
        return x


def DetectLan(img):
    lang_net = torch.load("Data/weights.pth")
    extractedInformation = pytesseract.image_to_string((img), lang='eng')
    print("Extracted Information (Tecceract default language):")
    print(extractedInformation)
    lan = mas[lang_net.inference(Take_exp(extractedInformation))]
    print("Detected language:",lan)
    extractedInformation = pytesseract.image_to_string((img), lang=lan)
    print(extractedInformation)
    return lan


def get_text(lan,image, t,l,b,r, file):
    file.write(' ' + pytesseract.image_to_string(im_cut(image, t,l,b,r), lang=lan))

