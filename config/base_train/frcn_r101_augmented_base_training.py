_base_ = [
    '../_base_/datasets/ft_augment.py',
    '../_base_/models/faster_rcnn_r50_caffe_fpn.py',
    '../_base_/base_settings.py',
]

model = dict(
    roi_head=dict(bbox_head=dict(num_classes=6)),
    )

data = dict(
    train=dict(classes='ALL_CLASSES_SPLIT1'),
    val=dict(classes='ALL_CLASSES_SPLIT1'),
    test=dict(classes='ALL_CLASSES_SPLIT1'))

# model settings
model = dict(
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(depth=101),
    roi_head=dict(bbox_head=dict(num_classes=6)),
    )
