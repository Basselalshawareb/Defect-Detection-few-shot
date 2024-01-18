# Copyright (c) OpenMMLab. All rights reserved.
try:
    import mmdet
except (ImportError, ModuleNotFoundError):
    mmdet = None

if mmdet is None:
    raise RuntimeError('mmdet is not installed')
import argparse
import os
import os.path as osp
import time
import warnings
#from mmfewshot.detection.models import build_detector
from mmdet.models import build_detector
import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet.apis import multi_gpu_test, single_gpu_test
#from mmfewshot.detection.apis import single_gpu_test
# from mmdet.datasets import (build_dataloader, build_dataset,
#                             replace_ImageToTensor)

from mmfewshot.detection.datasets import (build_dataloader, build_dataset,
                                          get_copy_dataset_type)

from mmrazor.models.builder import build_algorithm
from mmrazor.utils import setup_multi_processes
import pandas as pd
import numpy as np

from wrappers import SetTrainStage

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--augment',default='none', help='augment type')
    parser.add_argument('--work-dir',default=None, help='work dir')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    
    args = parser.parse_args()
    return args

def GetCheckPointPATH(checkpoint_folder):
    file_name_list = os.listdir(checkpoint_folder)
    for file_name in file_name_list:
        if "best_bbox_mAP" in file_name:
            best_ckpt_path = os.path.join(checkpoint_folder,file_name)
            return best_ckpt_path

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    # set multi-process settings
    setup_multi_processes(cfg)
    #SetTrainStage(cfg,"Pretrain_SPLIT1")
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    
    
    

    # currently only support single images testing
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    assert samples_per_gpu == 1, 'currently only support single images testing'

    # init distributed env first, since logger depends on the dist info.

    cfg.gpu_ids = range(1)
    
    
    if "algorithm" in cfg.keys():
    # build the algorithm and load checkpoint
        cfg.algorithm.architecture.model.pretrained = None
        if cfg.algorithm.architecture.model.get('neck'):
            if isinstance(cfg.algorithm.architecture.model.neck, list):
                for neck_cfg in cfg.algorithm.architecture.neck:
                    if neck_cfg.get('rfp_backbone'):
                        if neck_cfg.rfp_backbone.get('pretrained'):
                            neck_cfg.rfp_backbone.pretrained = None
            elif cfg.algorithm.architecture.model.neck.get('rfp_backbone'):
                if cfg.algorithm.architecture.model.neck.rfp_backbone.get(
                        'pretrained'):
                    cfg.algorithm.architecture.model.neck.rfp_backbone.pretrained = None  # noqa E501
        cfg.algorithm.architecture.model.train_cfg = None
        algorithm = build_algorithm(cfg.algorithm)
        model = algorithm.architecture.model
    else:
        cfg.model.pretrained = None
        model = build_detector(cfg.model)

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    
    checkpoint_path = GetCheckPointPATH(args.checkpoint)
    checkpoint = load_checkpoint(
        model, checkpoint_path, map_location='cpu')
    # if args.fuse_conv_bn:
    #     model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    model.CLASSES = checkpoint['meta']['CLASSES']

    model = MMDataParallel(model, device_ids=cfg.gpu_ids)
    test_data = cfg.data.test.copy()
    augment = args.augment
    augmentations = ['blur', 'bright', 'rotate']
    for augment in augmentations:
        if augment!='none':
            aug_type = augment
            aug_transform = []
            if 'blur' in aug_type:
                aug_transform += [dict(type='Blur', blur_limit=(10,15), p=1.0)]
            if 'bright' in aug_type:
                aug_transform += [ dict(
                    type='RandomBrightnessContrast',
                    brightness_limit=[-0.1,0.1],
                    contrast_limit=[-0.1,0.1],
                    p=1)]
            if 'rotat' in aug_type:
                aug_transform += [dict(
                    type='DkanRotate',
                    max_rotate_degree = 5,
                    scaling_ratio_range=(1, 1),
                    max_aspect_ratio = 100,
                    max_translate_ratio = 0.5,
                    max_shear_degree = 15,
                    skip_filter = False,
                    )]

            pip = cfg.data.train.pipeline
            for i,trans in enumerate(pip):
                if 'Albu' in trans['type']:
                    trans['transforms']=aug_transform
                    break


        # build the dataloader
        dataset = build_dataset(test_data)
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)
        outputs = single_gpu_test(model, data_loader, show_score_thr=0.3)

            
        out_file = osp.join(cfg.work_dir,"preds.pkl")
        print(f'\nwriting results to {out_file}')
        mmcv.dump(outputs, out_file)
        kwargs = {}
        # if args.format_only:
        #     dataset.format_results(outputs, **kwargs)

        eval_kwargs = cfg.get('evaluation', {}).copy()
        # hard-code way to remove EvalHook args
        for key in [
                'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                'rule', 'dynamic_intervals'
        ]:
            eval_kwargs.pop(key, None)
        eval_kwargs.update(dict(metric="bbox", **kwargs))
        metric = dataset.evaluate(outputs, **eval_kwargs)
        print(metric)
        # mAP_base = metric['BASE_CLASSES_SPLIT1 bbox_mAP']
        # mAP_novel = metric['NOVEL_CLASSES_SPLIT1 bbox_mAP']
        mAP = metric['bbox_mAP']
        table_path = "work_dirs/base/aug_table.csv"
        df = pd.read_csv(table_path,sep='&', index_col="index")
        df.loc[len(df.index)] = [aug_type,mAP]
        df.to_csv(table_path,sep='&')
if __name__ == '__main__':
    main()