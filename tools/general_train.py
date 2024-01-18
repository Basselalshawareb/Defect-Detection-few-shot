# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
from distutils.command.build import build
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist, set_random_seed
from mmcv.utils import get_git_hash
from mmdet.utils import collect_env
from mmdet.apis import single_gpu_test
from mmfewshot import __version__
from mmfewshot.detection.datasets import build_dataset
from mmfewshot.detection.models import build_detector
from mmfewshot.utils import get_root_logger

from mmrazor.models import build_algorithm
from mmrazor.utils import setup_multi_processes
from mmfewshot.detection.apis import train_detector
import pandas as pd
from wrappers import *

def parse_args():
    parser = argparse.ArgumentParser(description='Train a FewShot model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--ablation',default='logit')
    parser.add_argument('--alpha',default=20)        
    parser.add_argument(
        '--work-dir', default=None, help='the directory to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--work-flow',default=1, help='2 if train and test')
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

    if args.resume_from is not None:
        cfg.resume_from = args.resume_from


    cfg.gpu_ids = range(1)


    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    config_name = osp.basename(args.config)

    if "algorithm" in cfg.keys(): 
        assert args.ablation in ["none","feature","logit","both"]
        cfg = ablation_cfg(cfg,loss_ablation=args.ablation)
        cfg = ablation_cosine(cfg,alpha=args.alpha)
    save_name = f"{config_name[:-3]}-NEU_DET.py"
    cfg.dump(osp.join(cfg.work_dir, save_name))

    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Config:\n{cfg.pretty_text}')
    seed = None
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    # build_detector will do three things, including building model,
    # initializing weights and freezing parameters (optional).
    
    
    
    # build_dataset will do two things, including building dataset
    # and saving dataset into json file (optional).
    datasets = [
        build_dataset(
            cfg.data.train,
            rank=0,
            work_dir=cfg.work_dir,
            timestamp=timestamp)
    ]

    if len(cfg.workflow) == 2:
        data_test= copy.deepcopy(cfg.data.test)
        test_dataset = build_dataset(data_test)
        test_data_loader = build_dataloader(
        test_dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)
    
    if cfg.checkpoint_config is not None:
        # save mmfewshot version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmfewshot_version=__version__ + get_git_hash()[:7],
            CLASSES=datasets[0].CLASSES)


    if "algorithm" in cfg.keys(): 
        model = build_algorithm(cfg.algorithm)
        model.init_weights()
        model.CLASSES = datasets[0].CLASSES
        train_ifs_detector(
            model,
            datasets,
            cfg,
            distributed=False,
            validate=True,
            timestamp=timestamp,
            meta=meta
            )
    else:
        model = build_detector(cfg.model, logger=logger)
        model.CLASSES = datasets[0].CLASSES
        train_detector(
            model,
            datasets,
            cfg,
            distributed=False,
            validate=True,
            timestamp=timestamp,
            meta=meta
            )

    if len(cfg.workflow) == 2:

        checkpoint_path = GetCheckPointPATH(args.work_dir)
        checkpoint = load_checkpoint(
            model, checkpoint_path, map_location='cpu')
        
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)
        outputs = single_gpu_test(model, test_data_loader, show_score_thr=0.3)
        
        out_file = ""
        print(f'\nwriting results to {args.out}')
        mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule', 'dynamic_intervals'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            metric = dataset.evaluate(outputs, **eval_kwargs)
            print(metric)
            mAP_base = metric['BASE_CLASSES_SPLIT1 bbox_mAP']
            mAP_novel = metric['NOVEL_CLASSES_SPLIT1 bbox_mAP']
            mAP = metric['bbox_mAP']
            if args.name!='anything' and args.shot!='-1':
                csv_path = f"work_dirs/5shot/mAP_results_{args.shot}SHOT.csv"
                df = pd.read_csv(csv_path,index_col='MODEL')
                df.loc[args.name]=[mAP_base, mAP_novel, mAP]
                df.to_csv(csv_path)


if __name__ == '__main__':
    main()
