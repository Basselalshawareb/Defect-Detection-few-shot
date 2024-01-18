import os.path as osp

def SetTrainStage(cfg,setting ="SPLIT1_SEED1_10SHOT"):
    defect_data = "NEU_DET"
    assert defect_data in ["NEU_DET","DeepPCB","GC10_DET"]

    work_root = osp.dirname(__file__)
    data_root = f'{work_root}/../dataset/{defect_data}/'
    
    total_class_num = 6

    if "Pretrain" in setting:
    ### setting = Pretrain_SPLIT1
        split = setting.split("_")[-1][-1]       
        if cfg.data.train.type == "FewShotDefectDataset":
        ### for faster rcnn pretrain setting
            cfg.data.train.defect_name = defect_data
            cfg.data.val.defect_name = defect_data
            cfg.data.test.defect_name = defect_data

        else:
        ### for YOLO
            cfg.data.train.dataset.data_root = data_root
            cfg.data.train.dataset.defect_name = defect_data
            cfg.data.train.dataset.ann_cfg[0]["ann_file"]=f"annotations/trainval.json"
    else:
    ### setting = SPLIT1_SEED1_5SHOT
        if "AUGMENT" in setting:
            shot = int(setting.split("_")[-2][:-4])
            split = int(setting.split("_")[0][-1])
        else:
            shot = int(setting.split("_")[-1][:-4])
            split = int(setting.split("_")[0][-1])
        class_num = total_class_num

        cfg.data.val.data_root = data_root
        cfg.data.val.defect_name = defect_data
        cfg.data.val.ann_cfg[0]["ann_file"]=f"annotations/fine_tune/test.json"

        cfg.data.test.data_root = cfg.data.val.data_root 
        cfg.data.test.defect_name = defect_data
        cfg.data.test.ann_cfg[0]["ann_file"]=f"annotations/fine_tune/test.json"

        if cfg.data.train.type == "FewShotDefectDefaultDataset":
        ### TFA FSCE
            load_from_path = f"./weights/frcn_r101_split{split}_fine-tuning.pth"
            load_student = "./weights/faster_rcnn_r101_caffe_imagenet.pth"
            # if "algorithm" in cfg.keys():
            #     cfg.algorithm.architecture.model.init_cfg.checkpoint = load_from_path
            #     cfg.algorithm.distiller.teacher.init_cfg.checkpoint = load_from_path
            # else:
            #     cfg.load_from = load_from_path

            cfg.data.train.data_root = data_root
            cfg.data.train.ann_cfg[0]["dataset_name"]=defect_data
            cfg.data.train.ann_cfg[0]["setting"]=setting
            cfg.data.train.num_novel_shots = shot
            cfg.data.train.num_base_shots = shot
            cfg.data.train.classes=f'ALL_CLASSES_SPLIT{split}'

    return cfg    

def ablation_cfg(cfg,loss_ablation=None):
    assert loss_ablation in ["none","logit","feature","both"]

    distill_cfg = cfg.algorithm.distiller.components
    if loss_ablation == "logit":
        del distill_cfg[0]
    elif loss_ablation  == "feature":
        del distill_cfg[1:]
    elif loss_ablation == "both":
        del distill_cfg[:]    
    else:    
        return cfg
    return cfg

def ablation_cosine(cfg,alpha=20):
    alpha=int(alpha)
    if alpha ==0:
        cfg.algorithm.architecture.model.roi_head.bbox_head.type="Shared2FCBBoxHead"
        del cfg.algorithm.architecture.model.roi_head.bbox_head.scale
        del cfg.algorithm.architecture.model.roi_head.bbox_head.num_shared_fcs

    else: 
        cfg.algorithm.architecture.model.roi_head.bbox_head.scale=int(alpha)
    return cfg
