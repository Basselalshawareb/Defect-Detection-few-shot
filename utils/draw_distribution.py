#%%
NEU_Train = "/home/huemorgen/IFSDD/dataset/NEU_DET/annotations/trainval.json"
NEU_5shot = "/home/huemorgen/IFSDD/dataset/NEU_DET/annotations/fewshot-split/1/"

# NEU_Train = "/home/sunchen/Projects/SSDefect/dataset/GC10_DET/annotations/trainval.json"
# NEU_10shot = "/home/sunchen/Projects/SSDefect/dataset/GC10_DET/annotations/fewshot-split/1/"

import json
import os

def read_json(json_path):
    with open(json_path,"r") as load_f:
        json_file = json.load(load_f)
    return json_file

# def draw_bar(area_list):
#     area_thr = [32,64,128,256,512,1024]
#     # area_count = {thr:}
#     pass
    

NEU_JSON = read_json(NEU_Train)
NEU_anno = NEU_JSON["annotations"]
NEU_cat = NEU_JSON["categories"]
cat2area = {}
id2cat = {cat["id"]:cat["name"]for cat in NEU_cat}
print(id2cat)
for anno in NEU_anno:
    cat = anno["category_id"]
    if cat not in cat2area.keys():
        cat2area[cat]=[]
    cat2area[cat].append(anno["area"])
for key,value in cat2area.items():
    cat2area[key].sort()


import matplotlib.pyplot as plt
#plt.style.use(['ieee'])



for key in cat2area.keys():
    fig = plt.figure(dpi=200)
    catname = id2cat[key]
    fewshot_path = NEU_5shot+f"/1seed_5shot_{catname}_trainval.json"
    fewshot_json = read_json(fewshot_path)
    fewshot_anno = fewshot_json["annotations"]
    fs_area = [anno["area"] for anno in fewshot_anno]
    
    x = cat2area[key]
    
    plt.hist([x,fs_area],bins=10,label=['All Dataset', 'Few-shot Data'])
    plt.xlabel("Scale(pixel)")
    plt.ylabel("Number of Instances")
    plt.legend()
    plt.savefig(f"{catname}.png")
    plt.title(catname)
    plt.show()
# %%
