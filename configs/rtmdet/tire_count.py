_base_ = './rtmdet-ins_m_8xb32-300e_coco.py'

classes = ('tire',)  # change to ('tires', 'tire') if you really want 2 classes
num_classes = len(classes)
metainfo = dict(classes=classes)
data_root = 'data/tire_count/'

load_from = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_m_8xb32-300e_coco/rtmdet-ins_m_8xb32-300e_coco_20221123_001039-6eba602e.pth'
work_dir = 'work_dirs/rtmdet_ins_m_tire_count'

model = dict(
    bbox_head=dict(num_classes=num_classes),
    test_cfg=dict(score_thr=0.5, max_per_img=100),
)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=True,
    ),
    dict(type='Resize', scale=(640, 640), keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/_annotations.coco.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=1),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='valid/_annotations.coco.json',
        data_prefix=dict(img='valid/'),
        test_mode=True,
    ),
)
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='test/_annotations.coco.json',
        data_prefix=dict(img='test/'),
        test_mode=True,
    ),
)

val_evaluator = dict(
    ann_file=data_root + 'valid/_annotations.coco.json',
    metric=['bbox', 'segm'],
)
test_evaluator = dict(
    ann_file=data_root + 'test/_annotations.coco.json',
    metric=['bbox', 'segm'],
)

max_epochs = 10

train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=1,
)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.01),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
)

param_scheduler = []

default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=3,
        save_best='coco/segm_mAP',
        rule='greater',
    ),
)

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49,
    )
]

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
)

default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=3,
        save_best='coco/segm_mAP',
        rule='greater',
    ),
    logger=dict(type='LoggerHook', interval=10),
)