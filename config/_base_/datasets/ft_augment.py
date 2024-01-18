# dataset settings
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        keep_ratio=True,
        multiscale_mode='value'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type="Albu",
            transforms=[
                dict(type='Blur', blur_limit=(10,15), p=1.0),
                # dict(
                # type='RandomBrightnessContrast',
                # brightness_limit=[-0.1,0.1],
                # contrast_limit=[-0.1,0.1],
                # p=1)
            ],
            bbox_params=dict(
                type='BboxParams',
                format='pascal_voc',
                label_fields=['gt_labels'],
                filter_lost_elements = True,
                # filter_lost_elements = True,
                )
        ),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type="Albu_with_print",
                transforms=[
                    dict(type='Blur', blur_limit=(10,15), p=1.0),
                    # dict(
                    # type='RandomBrightnessContrast',
                    # brightness_limit=[-0.1,0.1],
                    # contrast_limit=[-0.1,0.1],
                    # p=1)
                ],
                bbox_params=None
                # dict(
                #     type='BboxParams',
                #     format='pascal_voc',
                #     label_fields=['gt_labels'],
                #     filter_lost_elements = True,
                #     # filter_lost_elements = True,
                #     )
            ),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
# classes splits are predefined in FewShotCocoDataset

data_root = "datasets/NEU_DET/"
annotations_root = "base/"
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='FewShotDefectDataset',
        save_dataset=False,
        data_root=data_root,
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file=annotations_root+'annotations/trainval.json')    
        ],
        img_prefix="images",
        pipeline=train_pipeline,
        classes="BASE_CLASSES_SPLIT1"),
    val=dict(
        type='FewShotDefectDataset',
        data_root=data_root,
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file=annotations_root+'annotations/test.json')    
        ],
        img_prefix="images",
        pipeline=test_pipeline,
        classes="BASE_CLASSES_SPLIT1"),
    test=dict(
        type='FewShotDefectDataset',
        data_root=data_root,
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file=annotations_root+'annotations/test.json')    
        ],
        img_prefix="images",
        pipeline=test_pipeline,
        test_mode=True,
        classes="BASE_CLASSES_SPLIT1"))
