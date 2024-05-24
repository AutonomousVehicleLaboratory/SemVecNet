_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]
#
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
# point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
point_cloud_range = [-30.0, -15.0, -5.0, 30.0, 15.0, 3.0]
voxel_size = [0.15, 0.15, 8.0]
dbound=[1.0, 35.0, 0.5]

grid_config = {
    'x': [-30.0, -30.0, 0.15], # useless
    'y': [-15.0, -15.0, 0.15], # useless
    'z': [-10, 10, 20],        # useless
    'depth': [1.0, 35.0, 0.5], # useful
}


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
# map has classes: divider, ped_crossing, boundary
map_classes = ['divider', 'ped_crossing','boundary', 'centerline']
# fixed_ptsnum_per_line = 20
# map_classes = ['divider',]
num_vec=70
fixed_ptsnum_per_gt_line = 20 # now only support fixed_pts > 0
fixed_ptsnum_per_pred_line = 20
eval_use_same_gt_sample_num_flag=True
num_map_classes = len(map_classes)

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 1
bev_h_ = 200
bev_w_ = 100
queue_length = 1 # each sequence contains `queue_length` frames.
modality = 'sem_map'

aux_seg_cfg = dict(
    use_aux_seg=False,
    bev_seg=True,
    pv_seg=False,
    seg_classes=1,
    feat_down_sample=32,
    pv_thickness=1,
)

z_cfg = dict(
    pred_z_flag=False,
    gt_z_flag=False,
)

model = dict(
    type='MapTRv2',
    use_grid_mask=True,
    video_test_mode=False,
    modality=modality,
    pretrained=dict(img='ckpts/resnet50-19c8e357.pth'),
    sem_map_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    sem_map_neck=dict(
        type='FPN',
        in_channels=[2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='MapTRv2Head',
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=900,
        num_vec_one2one=num_vec,
        num_vec_one2many=0,
        k_one2many=0,
        num_pts_per_vec=fixed_ptsnum_per_pred_line, # one bbox
        num_pts_per_gt_vec=fixed_ptsnum_per_gt_line,
        dir_interval=1,
        modality=modality,
        query_embed_type='instance_pts',
        transform_method='minmax',
        gt_shift_pts_pattern='v2',
        num_classes=num_map_classes,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        code_size=2,
        code_weights=[1.0, 1.0, 1.0, 1.0],
        aux_seg=aux_seg_cfg,
        z_cfg=z_cfg,
        transformer=dict(
            type='MapTRPerceptionTransformer',
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=False,
            embed_dims=_dim_,
            modality=modality,
            fuser=dict(
                type='ConvFuser',
                in_channels=[_dim_, 256],
                out_channels=_dim_,
            ),
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=1,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                        dict(
                            type='GeometrySptialCrossAttention',
                            pc_range=point_cloud_range,
                            attention=dict(
                                type='GeometryKernelAttention',
                                embed_dims=_dim_,
                                num_heads=4,
                                dilation=1,
                                kernel_size=(3,5),
                                num_levels=_num_levels_),
                            embed_dims=_dim_,
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            decoder=dict(
                type='MapTRDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DecoupledDetrTransformerDecoderLayer',
                    num_vec=num_vec,
                    num_pts_per_vec=fixed_ptsnum_per_pred_line,
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                         dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                    ],

                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'self_attn', 'norm','cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='MapTRNMSFreeCoder',
            # post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            z_cfg=z_cfg,
            post_center_range=[-35, -20, -35, -20, 35, 20, 35, 20],
            pc_range=point_cloud_range,
            max_num=50,
            voxel_size=voxel_size,
            num_classes=num_map_classes),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
            ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.0),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),
        loss_pts=dict(type='PtsL1Loss', 
                      loss_weight=5.0),
        loss_dir=dict(type='PtsDirCosLoss', loss_weight=0.005),
        loss_seg=dict(type='SimpleLoss', 
            pos_weight=4.0,
            loss_weight=1.0),
        loss_pv_seg=dict(type='SimpleLoss', 
                    pos_weight=1.0,
                    loss_weight=2.0),),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='MapTRAssigner',
            z_cfg=z_cfg,
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=0.0, box_format='xywh'),
            # reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            # iou_cost=dict(type='IoUCost', weight=1.0), # Fake cost. This is just to make it compatible with DETR head.
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=0.0),
            pts_cost=dict(type='OrderedPtsL1Cost', 
                      weight=5),
            pc_range=point_cloud_range))))

dataset_type = 'CustomAV2SemanticMapDataset'
data_root = '/cogrob-avl-dataset/argoverse2/sensor/'
file_client_args = dict(backend='disk')


train_pipeline = [
    dict(type='CustomLoadSemanticMapFromFile', to_float32=True),
    dict(type='NormalizeSemanticMap', **img_norm_cfg),
    dict(type='CustomMapDefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['sem_map'])
]

test_pipeline = [
    dict(type='CustomLoadSemanticMapFromFile', to_float32=True),
    dict(type='NormalizeSemanticMap', **img_norm_cfg),
    dict(type='CustomMapDefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='CustomCollect3D', keys=['sem_map'])
]

data = dict(
    samples_per_gpu=5,
    workers_per_gpu=5,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'av2_map_infos_train.pkl',
        z_cfg=z_cfg,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        aux_seg=aux_seg_cfg,
        test_mode=False,
        use_valid_flag=True,
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
        eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
        padding_value=-10000,
        map_classes=map_classes,
        queue_length=queue_length,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'av2_map_infos_val.pkl',
        map_ann_file=data_root + 'av2_gt_map_anns_val.json',
        # code_size=3,
        z_cfg=z_cfg,
        load_interval=4, # av2 uses 10 Hz, set to 5, 2HZ the same as nuscenes,
        # load_interval=1, # TODO debug
        pipeline=test_pipeline,  
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
        eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
        padding_value=-10000,
        map_classes=map_classes,
        classes=class_names, 
        modality=input_modality, 
        samples_per_gpu=1),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'av2_map_infos_val.pkl',
        map_ann_file=data_root + 'av2_gt_map_anns_val.json',
        # code_size=3,
        z_cfg=z_cfg,
        load_interval=4, # av2 uses 10 Hz, set to 5, 2HZ the same as nuscenes,
        # load_interval=1, # TODO debug
        pipeline=test_pipeline, 
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
        eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
        padding_value=-10000,
        map_classes=map_classes,
        classes=class_names, 
        modality=input_modality),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW',
    lr=6e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
total_epochs = 6
evaluation = dict(interval=3, pipeline=test_pipeline, metric='chamfer')
# total_epochs = 50
# evaluation = dict(interval=1, pipeline=test_pipeline)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# fp16 = dict(loss_scale=512.)
checkpoint_config = dict(interval=1)
find_unused_parameters=True
