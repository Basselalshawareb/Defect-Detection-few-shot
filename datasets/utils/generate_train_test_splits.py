import json
import os.path as osp
import argparse
from sklearn.model_selection import train_test_split
import random
CLASSES = {
            1: "crazing", 2: "inclusion", 3: "patches",
            4: "pitted_surface", 5: "rolled-in_scale", 6: "scratches",
        }

CATS = [dict({"supercategory":"None","id":i, "name":value }) for i, value in CLASSES.items()]

def split(pth,train_ratio=0.1, mode="num_samples",num_samples = 80):
    file = json.load(open(pth))
    anns = file['annotations']
    images = file['images']
    
    ann_cats = dict()
    for i in range(1,len(CLASSES.values())+1):
        ann_cats.update({i:[]})
    
    
    for i,ann in enumerate(anns):
        ann_cats[ann['category_id']].append(i)
    
    train_test_anns = {"train":[], "test":[]}
    for cat_ls in ann_cats.values():
        if mode=="num_samples":
            all_samples = cat_ls.copy()
            random.shuffle(all_samples)
            train = all_samples[:num_samples]
            test = all_samples[num_samples:]
        else:
            train, test = train_test_split(cat_ls,train_size = train_ratio, shuffle=True)
        train_test_anns['train']= train_test_anns['train'] + [anns[id] for id in train]
        train_test_anns['test']= train_test_anns['test'] + [anns[id] for id in test]
    random.shuffle(train_test_anns['train'])
    random.shuffle(train_test_anns['test'])
    train_image_ids = set()
    for ann in train_test_anns['train']:
        train_image_ids.add(ann['image_id'])
    test_image_ids = set()
    for ann in train_test_anns['test']:
        test_image_ids.add(ann['image_id'])

    images_out = dict({"train":[],"test":[]})

    for img in images:
        if not ".jpg" in img['file_name']:
            img['file_name']+=".jpg"
        id = img['id']
        if id in train_image_ids:
            images_out["train"].append(img)
        if id in test_image_ids:
            images_out["test"].append(img)
    return train_test_anns['train'],train_test_anns['test'],images_out["train"],images_out["test"]
    
def save(anns_train, anns_test, images_train, images_test, save_directory):
    
    new_data = dict({"images":images_train,"annotations":anns_train,"categories":CATS})
    with open(f"{save_directory}/train.json", 'w') as f:
        json.dump(new_data, f,indent=4, separators=(',', ': '))
    
    new_data = dict({"images":images_test,"annotations":anns_test,"categories":CATS})
    with open(f"{save_directory}/test.json", 'w') as f:
        json.dump(new_data, f,indent=4, separators=(',', ': '))

if __name__=="__main__":

    pth = "datasets/NEU_DET/all_samples.json"
    save_directory = "/home/huemorgen/DefectDetection/datasets/NEU_DET/traintest"
    anns_train, anns_test, images_train, images_test = split(pth, train_ratio=0.1)
    save(anns_train, anns_test, images_train, images_test,save_directory)