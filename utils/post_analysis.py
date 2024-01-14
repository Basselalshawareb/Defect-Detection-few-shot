import os
import os.path as osp
import time
import warnings
import json
import mmcv
import torch
from torch.distributed import launch
from mmcv import Config
from mmcv.parallel import MMDataParallel,MMDistributedDataParallel

import sys
path = osp.dirname(osp.abspath(__file__))+"/../"
sys.path.append(path)
from dkan import *
from analysis import ResultVisualizer,bbox_map_eval
from mmdet.datasets import get_loading_pipeline
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.utils import setup_multi_processes
import numpy as np
import pickle

def SetDefectDataset(cfg,defect_data="NEU_DET",setting ="10SHOT_SEED1"):
    assert defect_data in ["NEU_DET","DeepPCB","GC10_DET"]
    assert "SHOT_SEED" in setting
    print(defect_data)
    if cfg.model.type =='MPSR':
        cfg.data.train.dataset.ann_cfg[0]["dataset_name"]=defect_data
        cfg.data.train.dataset.ann_cfg[0]["setting"]=setting
        cfg.data.train.dataset.img_prefix = f'/home/sunchen/Projects/SSDefect/dataset/{defect_data}/images'
        cfg.data.train.auxiliary_dataset["defect_name"]=defect_data
    else:
        cfg.data.train.ann_cfg[0]["dataset_name"]=defect_data
        cfg.data.train.ann_cfg[0]["setting"]=setting
        cfg.data.train.img_prefix = f'/home/sunchen/Projects/SSDefect/dataset/{defect_data}/images'

    cfg.data.val.data_root = f'/home/sunchen/Projects/SSDefect/dataset/{defect_data}/'
    cfg.data.val.img_prefix = f'/home/sunchen/Projects/SSDefect/dataset/{defect_data}/images'
    cfg.data.val.defect_name = defect_data

    cfg.data.test.data_root = cfg.data.val.data_root 
    cfg.data.test.img_prefix = cfg.data.val.img_prefix
    cfg.data.test.defect_name = defect_data
    
    return cfg

def bbox_visualize(cfg,outputs,show_dir,gt_only=False):
    if isinstance(outputs,str):
        outputs=mmcv.load(outputs)

    cfg.data.test.test_mode = True
    os.makedirs(show_dir,exist_ok=True)
    cfg.data.test.pop('samples_per_gpu', 0)

    if cfg.model.type=="MPSR":
        cfg.data.test.pipeline = get_loading_pipeline(cfg.data.train.dataset.multi_pipelines.main) 
    else:
        cfg.data.test.pipeline = get_loading_pipeline(cfg.data.train.pipeline)
        
    datasets = build_dataset(cfg.data.test)
    print(datasets)
    result_visualizer = ResultVisualizer(score_thr=0.3)
    if gt_only:
        result_visualizer._save_img_gt(datasets,show_dir)
    else:
        prog_bar = mmcv.ProgressBar(len(outputs))
        _mAPs = {}
        for i, (result, ) in enumerate(zip(outputs)):
            # self.dataset[i] should not call directly
            # because there is a risk of mismatch
            data_info = datasets.prepare_train_img(i)
            mAP = bbox_map_eval(result, data_info['ann_info'])
            _mAPs[i] = mAP
            prog_bar.update()

        # descending select topk image
        _mAPs = list(sorted(_mAPs.items(), key=lambda kv: kv[1]))
        # print(_mAPs)
        result_visualizer._save_image_gts_results(datasets, outputs, _mAPs, show_dir)


if __name__=="__main__":

    for data in ["NEU_DET"]:#["DeepPCB","GC10_DET"]:
        # cfg_path = f"/home/sunchen/Projects/SSDefect/work_dir/20230301/{data}/FCOS/5SHOT/FCOS-{data}-5SHOT_SEED1.py"
        # cfg = Config.fromfile(cfg_path)
        # show_dir = f"/home/sunchen/Projects/SSDefect/visual/{data}-gt"

        # bbox_visualize(cfg,outputs=None,show_dir=show_dir,gt_only=True)
        for shot in 5,10,30:
            # for model in ['YOLOV3','FCOS','Retina','FRCN']:
            for model in ["MPSR"]:
                output=f"/home/sunchen/Projects/SSDefect/work_dir/20230226/{data}/{model}/{shot}SHOT/output.pkl"
                cfg_path = f"/home/sunchen/Projects/SSDefect/work_dir/20230226/{data}/{model}/{shot}SHOT/{model}-{data}-{shot}SHOT_SEED1.py"
                cfg = Config.fromfile(cfg_path)
                cfg = SetDefectDataset(cfg,defect_data=data,setting=f"{shot}SHOT_SEED1")
                show_dir = f"/home/sunchen/Projects/SSDefect/visual/{data}/{shot}/{model}"
                bbox_visualize(cfg,outputs=output,show_dir=show_dir,gt_only=False)
