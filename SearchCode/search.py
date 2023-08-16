#coding = utf-8
import numpy as np
import urllib.request
from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
import inflect
import time
import os
import shutil
import string
import re
import cv2
import argparse

import spacy
from itertools import combinations

parser = argparse.ArgumentParser(description='')
parser.add_argument('--k', type=int, default=1)
parser.add_argument('--dataset', type=str, default='voc', choices=['voc','coco'])
parser.add_argument('--synonym_file', type=str, default='voc_synonym.txt')
parser.add_argument('--store_path', type=str, default='testimages')
parser.add_argument('--img_num', type=int, default=5)
parser.add_argument('--check_img_num', type=int, default=20)

label_voc = ['person', 
            'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 
            'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', 
            'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tv']

super_category_voc = ['person', 
                    'animal', 'animal', 'animal', 'animal', 'animal', 'animal',
                    'vehicle', 'vehicle', 'vehicle', 'vehicle', 'vehicle', 'vehicle', 'vehicle',
                    'indoor', 'indoor', 'indoor', 'indoor', 'indoor', 'indoor']

# hair drier -> hair dryer
label_coco = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair dryer", "toothbrush"]

super_category_coco = ['person', 'vehicle', 'vehicle', 'vehicle', 'vehicle', 'vehicle', 'vehicle', 'vehicle', 'vehicle', 'outdoor', 'outdoor', 'outdoor', 'outdoor', 'outdoor', 'animal', 'animal', 'animal', 'animal', 'animal', 'animal', 'animal', 'animal', 'animal', 'animal', 'accessory', 'accessory', 'accessory', 'accessory', 'accessory', 'sports', 'sports', 'sports', 'sports', 'sports', 'sports', 'sports', 'sports', 'sports', 'sports', 'kitchen', 'kitchen', 'kitchen', 'kitchen', 'kitchen', 'kitchen', 'kitchen', 'food', 'food', 'food', 'food', 'food', 'food', 'food', 'food', 'food', 'food', 'furniture', 'furniture', 'furniture', 'furniture', 'furniture', 'furniture', 'electronic', 'electronic', 'electronic', 'electronic', 'electronic', 'electronic', 'appliance', 'appliance', 'appliance', 'appliance', 'appliance', 'indoor', 'indoor', 'indoor', 'indoor', 'indoor', 'indoor', 'indoor']

nlp = spacy.load("en_core_web_lg")
p = inflect.engine()

def search(lc, label, lc_syn, keyword, img_num, max_num, store_path):
    lc_str = '-'.join([label[i] for i in lc])
    path=os.path.join(store_path, lc_str)
    if not os.path.exists(path):
        os.makedirs(path)
        count = 0
    else:
        count = len(os.listdir(path))
        if count >= img_num:
            return   
    try:

        service = Service('./chromedriver.exe')
        browser = webdriver.Chrome(service=service)
        browser.get('https://www.google.com/imghp?hl=en');
        browser.maximize_window()
        browser.find_element(By.CLASS_NAME,'gLFyf').send_keys(keyword)
        browser.find_element(By.CLASS_NAME,'Tg7LZd').click()
        images = browser.find_elements(By.CLASS_NAME,'rg_i.Q4LuWd')
        if images.__len__() < max_num:
            # click show more images
            js = "window.scrollBy(0,2000)"
            browser.execute_script(js)
            images = browser.find_elements(By.CLASS_NAME, 'rg_i.Q4LuWd')
        images = images[:max_num]
        actions = ActionChains(browser)
        for img in images:
            if count >= img_num:
                return
            src = img.get_attribute('src')
            title = img.get_attribute('alt').strip()
            if src !=  None:
                # delete extra space
                title = re.sub('\s+',' ',title)
                # delete character not {letter, num, }
                myre = re.compile("[^0-9A-Za-z\s"+string.punctuation+"]")
                title = myre.sub('', title)
                # delete extra space
                title = re.sub('\s+',' ',title)
                if contain_key(lc_syn, title):
                    actions.click(img)
                    actions.perform()
                    time.sleep(1)
                    #highqualityimg = browser.find_elements(By.CLASS_NAME, 'n3VNCb.KAlRDb')
                    highqualityimg = browser.find_elements(By.XPATH, '//*[@id="Sva75c"]/div[2]/div[2]/div[2]/div[2]/c-wiz/div/div/div/div[3]/div[1]/a/img[1]')
                    src = highqualityimg[0].get_attribute('src').strip()
                    opener=urllib.request.build_opener()
                    opener.addheaders=[('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
                    urllib.request.install_opener(opener)
                    urllib.request.urlretrieve(src, os.path.join(path, lc_str + '_' + str(count+1) + ".png"))
                    # check whether is repeated image
                    if image_repeated(path, lc_str, count+1):
                        os.remove(os.path.join(path, lc_str + '_' + str(count+1) + ".png"))
                    count += 1
                    print(count)
        browser.quit()
    except Exception as e:
        print(e)

def image_repeated  (path, label2, count):
    image = cv2.imread(os.path.join(path, label2 + '_' + str(count) + ".png"))
    for i in range(1,count):
        downloaded_image = cv2.imread(os.path.join(path, label2 + '_' + str(i) + ".png"))
        if image.shape != downloaded_image.shape:
            continue
        difference = cv2.subtract(image, downloaded_image)    
        repeated = not np.any(difference)
        if repeated == True:
            cv2.destroyAllWindows() 
            return True
    cv2.destroyAllWindows()
    return False

def contain_key(lc, title):
    title = title.lower()
    doc = nlp(title)
    label_single = []
    label_multi = []
    doc_multi = []
    for label in lc:
        if len(label.split(' ')) > 1:
            label_multi.append(label)
            doc_multi.append(nlp(label))
        else:
            label = nlp(label)[0].lemma_
            label_single.append(label) 

    valid = [1]*len(doc)

    for i in range(len(doc)):
        # spacy would sperate words joint by hyphen
        # label occurs in compound word is invalid
        if doc[i].lemma_=='-':
            valid[i] = 0
            if i-1>=0:
                valid[i-1] = 0
            if i+1<len(doc):
                valid[i+1] = 0
    
        # labels appear consecutively is invalid (1 1)
        if i+1 < len(doc):
            if doc[i].lemma_ in label_single and doc[i+1].lemma_ in label_single:
                valid[i] = 0
                valid[i+1] = 0
        
        for index, label in enumerate(label_multi):
            label_word_num = len(label.split(' '))
            if i+label_word_num < len(doc):
                # labels appear consecutively is invalid (1 m)
                if doc[i].lemma_ in label_single:
                    exists = True
                    for j in range(label_word_num):
                        if doc[i+j+1].lemma_ != doc_multi[index][j].lemma_:
                            exists = False
                    if exists:
                        for j in range(i,i+label_word_num+1):
                            valid[j] = 0
                # labels appear consecutively is invalid (1 m)
                if doc[i+label_word_num].lemma_ in label_single:
                    exists = True
                    for j in range(label_word_num):
                        if doc[i+j].lemma_ != doc_multi[index][j].lemma_:
                            exists = False
                    if exists:
                        for j in range(i,i+label_word_num+1):
                            valid[j] = 0     
    
            # labels appear consecutively is invalid (m m)
            for index1 in range(len(label_multi)):
                for index2 in range(index+1, len(label_multi)):
                    num1, num2 = len(label_multi[index1].split(' ')), len(label_multi[index2].split(' '))
                    if i + num1 + num2 -1 < len(doc):
                        exists = True
                        for j in range(num1):
                            if doc[i+j].lemma_!=doc_multi[index1][j].lemma_:
                                exists = False
                        for j in range(num2):
                            if doc[i+num1+j].lemma_!=doc_multi[index2][j].lemma_:
                                exists = False
                        if exists:
                            for j in range(i,i+num1+num2):
                                valid[j] = 0

                        exists = True
                        for j in range(num2):
                            if doc[i+j].lemma_!=doc_multi[index2][j].lemma_:
                                exists = False
                        for j in range(num1):
                            if doc[i+num2+j].lemma_!=doc_multi[index1][j].lemma_:
                                exists = False
                        if exists:
                            for j in range(i,i+num1+num2):
                                valid[j] = 0


    count = 0
    for label in label_single:
        for i in range(len(doc)):
            token = doc[i]
            if valid[i] == 1:
                if token.lemma_==label and token.pos_=='NOUN':
                    count += 1
                    break

    for i in range(len(label_multi)):
        num = len(label_multi[i].split(' '))
        for j in range(len(doc)-num+1):
            t = 0
            for k in range(num):
                if valid[j+k] == 1 and doc[j+k].lemma_==doc_multi[i][k].lemma_:
                    t += 1
            if t==num:
                count += 1
                break
        
    if count == len(lc):
        return True
    else:
        return False
    
def get_syn(filename):
    synsets = {}
    with open(filename) as f:
        for line in f.readlines():
            a = eval(line.split('\t')[1].strip())
            a.insert(0,line.split('\t')[0].strip())
            synsets[line.split('\t')[0].strip()] = a
    return synsets


def get_syns_keywords(label, super_category, lc, syns):
    k = len(lc)
    keywords = []
    lc_syns = []
    recur(0, k, label, syns, lc, list(lc), lc_syns)
    for lc_syn in lc_syns:
        keyword = ''
        words = []
        for i in range(k):
            words.extend(lc_syn[i].split())
            if label[lc[i]] == super_category[lc[i]]:
                keyword += p.a(lc_syn[i])
            else:
                keyword += p.a(super_category[lc[i]]) + ' ' + lc_syn[i]
            if i!=k-1:
                keyword += ' AND '
        for word in words:
            keyword += ' intitle:' + word
        keyword += ' inurl:photo'
        keywords.append(keyword)    
    return lc_syns, keywords

def recur(i, k, label, syns, lc, lc_syn, lc_syns):
    if i==k:
        temp_list = lc_syn[:]
        lc_syns.append(temp_list)
    else:
        for syn in syns[label[lc[i]]]:
            lc_syn[i] = syn
            recur(i+1, k, label, syns, lc, lc_syn, lc_syns)

def main():
    args = parser.parse_args()
    dataset = args.dataset
    synonymfile = args.synonym_file
    k = args.k
    store_path = args.store_path
    img_num = args.img_num
    max_num = args.check_img_num
    
    if dataset=='voc':
        label = label_voc
        super_category = super_category_voc
    else:
        label = label_coco
        super_category = super_category_coco
    
    syns = get_syn(synonymfile)

    for lc in combinations(range(len(label)), k):
        lc_syns, keywords = get_syns_keywords(label, super_category, lc, syns)
        for i in range(len(keywords)):
            lc_syn, keyword = lc_syns[i], keywords[i]
            search(lc, label, lc_syn, keyword, img_num, max_num, store_path)
        keyword_simple = ' AND '.join([label[x] if label[x]==super_category[x] else label[x] + ' (' + super_category[x]+ ')' for x in lc])
        search(lc, label, list([label[x] for x in lc]), keyword_simple, img_num, max_num, store_path)   

if __name__ == '__main__':
    main()
