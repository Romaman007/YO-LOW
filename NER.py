import spacy

import os

def NER(ff):
    nlp=spacy.load("Ner1")
    print(nlp.pipe_names)

    with open("temp1.txt",encoding="UTF-8") as file:
        f = file.read()
        #ff = open("temp2.txt",'w',encoding="UTF-8")
        print(f)
    doc=nlp(f)
    for ent in doc.ents:
        ff.write("    "+ent.text+" "+ent.label_+",\n")
        print(ent.text,ent.label_)
    # path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'temp1.txt')
    # os.remove(path)
    # path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'resized_image1.jpg')
    # os.remove(path)