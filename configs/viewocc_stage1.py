"""Stage-one ViewOcc training with Plucker FiLM supervision."""

_base_ = ['./_base_/viewocc_base.py']

target_point_cloud_range = [-25, -25, -5.0, 25, 25, 3.0]
target_occ_size = [100, 100, 16]
occ_size = target_occ_size

model = dict(
    type='ViewOcc',
    pts_bbox_head=dict(
        volume_h=[100, 50, 25],
        volume_w=[100, 50, 25],
        volume_z=[16, 8, 4],
        conv_input=[512, 256, 256, 128, 128],
        conv_output=[256, 256, 128, 128, 64],
        out_indices=[0, 2, 4],
        upsample_strides=[1, 2, 1, 2, 1],
        volume_embedding_load_src_shapes=[
            [100, 100, 8],
            [50, 50, 4],
            [25, 25, 2],
        ],
        volume_embedding_load_src_pc_range=[
            -50, -50, -5.0, 50, 50, 3.0],
        transformer_template=dict(
            encoder=dict(
                pc_range=target_point_cloud_range,
                transformerlayers=dict(
                    attn_cfgs=[
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=target_point_cloud_range,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=[128, 256, 512],
                                num_points=[2, 4, 8],
                                num_levels=1),
                            embed_dims=[128, 256, 512],
                        )
                    ],
                ),
            ),
        ),
    ),
    plucker_raymap_cfg=dict(
        image_size=(640, 640),
        intrinsic=((320.0, 0.0, 320.0),
                   (0.0, 320.0, 320.0),
                   (0.0, 0.0, 1.0)),
        num_cams=6,
        pixel_center=True,
        normalize_dirs=True,
        origin_scale=100.0),
    plucker_fusion_channels=(512, 512, 512),
    plucker_hidden_channels=64,
    plucker_film_gamma_init=(0.15, 0.10, 0.05),
    plucker_film_beta_init=(0.10, 0.08, 0.05),
    ray_consistency_cfg=dict(loss_weight=0.0),
)

data = dict(
    samples_per_gpu=2,
    train=dict(
        ann_file=[
            '<TRAIN_SOURCE_ANNOTATIONS>',
            '<TRAIN_TARGET_ANNOTATIONS>',
        ],
        pc_range=target_point_cloud_range,
        occ_size=target_occ_size,
    ),
    val=dict(
        ann_file='<VALIDATION_ANNOTATIONS>',
        pc_range=target_point_cloud_range,
        occ_size=target_occ_size,
    ),
    test=dict(
        ann_file='<TEST_ANNOTATIONS>',
        pc_range=target_point_cloud_range,
        occ_size=target_occ_size,
    ),
)

load_from = '<PRETRAINED_CHECKPOINT>'
work_dir = '<STAGE1_WORK_DIR>'

evaluation = dict(
    interval=1,
    save_best='mIoU',
    rule='greater',
)

checkpoint_config = None

runner = dict(type='EpochBasedRunner', max_epochs=36)
