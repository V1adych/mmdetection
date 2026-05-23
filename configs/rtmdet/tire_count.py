_base_ = "./rtmdet-ins_m_8xb32-300e_coco.py"

classes = ("tire",)  # change to ('tires', 'tire') if you really want 2 classes
num_classes = len(classes)
metainfo = dict(classes=classes)
data_root = "data/tire_count/"

load_from = "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_m_8xb32-300e_coco/rtmdet-ins_m_8xb32-300e_coco_20221123_001039-6eba602e.pth"
work_dir = "work_dirs/rtmdet_ins_m_tire_count"

model = dict(
    bbox_head=dict(num_classes=num_classes),
    test_cfg=dict(score_thr=0.5, max_per_img=100),
)

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args={{_base_.backend_args}}),
    dict(
        type="LoadAnnotations",
        with_bbox=True,
        with_mask=True,
        poly2mask=True,
    ),
    dict(
        type="Albu",
        transforms=[
            dict(type="Affine", translate_percent=(-0.05, 0.05), scale=(0.75, 1.25), rotate=(-20, 20), fit_output=False, p=0.5),
            dict(
                type="OneOf",
                transforms=[
                    dict(type="Blur", blur_limit=5, p=1.0),
                    dict(type="MotionBlur", blur_limit=5, p=1.0),
                    dict(type="MedianBlur", blur_limit=5, p=1.0),
                ],
                p=0.25,
            ),
            dict(type="RandomBrightnessContrast", brightness_limit=0.2, contrast_limit=0.2, p=0.15),
            dict(type="GaussNoise", std_range=(0.04, 0.12), mean_range=(0.0, 0.0), p=0.15),
            dict(
                type="CoarseDropout",
                num_holes_range=(1, 3),
                hole_height_range=(0.03, 0.10),
                hole_width_range=(0.03, 0.10),
                fill=114,
                fill_mask=0,
                p=0.15,
            ),
        ],
        bbox_params=dict(
            type="BboxParams",
            format="pascal_voc",
            label_fields=["gt_bboxes_labels"],
            min_visibility=0.0,
            filter_lost_elements=True,
        ),
        keymap=dict(img="image", gt_bboxes="bboxes", gt_masks="masks"),
        skip_img_without_anno=True,
    ),
    dict(type="Resize", scale=(640, 640), keep_ratio=False),
    dict(type="RandomFlip", prob=0.5, direction="horizontal"),
    dict(type="RandomFlip", prob=0.5, direction="vertical"),
    dict(type="PackDetInputs"),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file="train/_annotations.coco.json",
        data_prefix=dict(img="train/"),
        filter_cfg=dict(filter_empty_gt=True, min_size=1),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file="valid/_annotations.coco.json",
        data_prefix=dict(img="valid/"),
        test_mode=True,
    ),
)
test_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file="test/_annotations.coco.json",
        data_prefix=dict(img="test/"),
        test_mode=True,
    ),
)

val_evaluator = dict(
    ann_file=data_root + "valid/_annotations.coco.json",
    metric=["bbox", "segm"],
)
test_evaluator = dict(
    ann_file=data_root + "test/_annotations.coco.json",
    metric=["bbox", "segm"],
)

max_epochs = 10

train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=1,
)

optim_wrapper = dict(
    _delete_=True,
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=1e-4, weight_decay=0.01),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
)

param_scheduler = []

default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=3,
        save_best="coco/segm_mAP",
        rule="greater",
    ),
)

custom_hooks = [
    dict(
        type="EMAHook",
        ema_type="ExpMomentumEMA",
        momentum=0.0002,
        update_buffers=True,
        priority=49,
    )
]

vis_backends = [
    dict(type="LocalVisBackend"),
    dict(type="TensorboardVisBackend"),
]

visualizer = dict(
    type="DetLocalVisualizer",
    vis_backends=vis_backends,
    name="visualizer",
)

default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=3,
        save_best="coco/segm_mAP",
        rule="greater",
    ),
    logger=dict(type="LoggerHook", interval=10),
)
