# SuperTickets: Drawing Task-Agnostic Lottery Tickets from Supernets via Jointly Architecture Searching and Parameter Pruning

**Haoran You**, Baopu Li, Zhanyi Sun, Xu Ouyang, Yingyan Lin

Submitted to ECCV 2022

## Environment

Require Python3, CUDA>=10.1, and torch>=1.4, all dependencies are as follows:

```shell script
pip3 install torch==1.4.0 torchvision==0.5.0 opencv-python tqdm tensorboard lmdb pyyaml packaging Pillow==6.2.2 matplotlib yacs pyarrow==0.17.1
pip3 install cityscapesscripts  # for Cityscapes segmentation
pip3 install mmcv-full # for Segmentation data loader
pip3 install pycocotools shapely==1.6.4 Cython pandas pyyaml json_tricks scikit-image  # for COCO keypoint estimation
```
or ```conda env create -f environment.yml```

## Setup

> Training script in general

```bash
python3 -m torch.distributed.launch --nproc_per_node=${NPROC_PER_NODE} --nnodes=${N_NODES} \
        --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
        --use_env train.py app:configs/YOUR_TASK.yml
```

Supported tasks:
- cls_imagenet
- seg_cityscapes
- seg_ade20k
- keypoint_coco

> Examples for training SuperTickets on ImageNet, COCO keypoint, Cityscapes, and ADE20K

```bash
python3 -m torch.distributed.launch \
--nproc_per_node=8 \
--nnodes=1 \
--node_rank=0 \
--master_port=1234 \
--use_env train.py app:configs/cls_imagenet_g_0.9.yml
```
```bash
python3 -m torch.distributed.launch \
--nproc_per_node=8 \
--nnodes=1 \
--node_rank=0 \
--master_port=1234 \
--use_env train.py app:configs/keypoint_coco_g_0.9.yml
```
```bash
python3 -m torch.distributed.launch \
--nproc_per_node=8 \
--nnodes=1 \
--node_rank=0 \
--master_port=1234 \
--use_env train.py app:configs/seg_cityscapes_g_0.8.yml
```
```bash
python3 -m torch.distributed.launch \
--nproc_per_node=8 \
--nnodes=1 \
--node_rank=0 \
--master_port=1234 \
--use_env train.py app:configs/seg_ade20k_g_0.9.yml
```

> Examples for testing the trained models

```bash
python3 -m torch.distributed.launch \
--nproc_per_node=8 \
--nnodes=1 \
--node_rank=0 \
--master_port=1234 \
--use_env validate.py app:configs/cls_imagenet_g_0.9.yml \
--resume 'output/cls_imagenet_g_0.9/20220202-001816/pruned_checkpoint.pt' \
--test_only True
```
```bash
python3 -m torch.distributed.launch \
--nproc_per_node=8 \
--nnodes=1 \
--node_rank=0 \
--master_port=1234 \
--use_env validate.py app:configs/keypoint_coco_g_0.9.yml \
--resume 'output/keypoint_coco_g_0.9/20220211-160107/pruned_checkpoint.pt' \
--test_only True
```
```bash
python3 -m torch.distributed.launch \
--nproc_per_node=8 \
--nnodes=1 \
--node_rank=0 \
--master_port=1234 \
--use_env validate.py app:configs/seg_cityscapes_g_0.8.yml \
--resume 'output/seg_cityscapes_g_0.7/20220108-175942/pruned_checkpoint.pt' \
--test_only True
```
```bash
python3 -m torch.distributed.launch \
--nproc_per_node=8 \
--nnodes=1 \
--node_rank=0 \
--master_port=1234 \
--use_env validate.py app:configs/seg_ade20k_g_0.9.yml \
--resume 'output/seg_ade20k_g_0.9/20220119-223955/pruned_checkpoint.pt' \
--test_only True
```

> Example for transferring from ImageNet to COCO keypoint
> (only transfer the feature extractor while keep the classification layer unchanged)

```bash
python3 -m torch.distributed.launch \
--nproc_per_node=8 \
--nnodes=1 \
--node_rank=0 \
--master_port=1234 \
--use_env transfer_img.py app:configs/transfer/cls_imagenet_to_keypoint_coco_g_0.9.yml \
--pretrained "output/cls_imagenet_g_0.9/20220209-130559/retrain_checkpoint.pt" \
--cur_model 'output/keypoint_coco_g_0.9/20220211-160107/pruned_checkpoint.pt'
```

## Datasets

1. ImageNet
    - Prepare ImageNet data following pytorch [example](https://github.com/pytorch/examples/tree/master/imagenet).
    - Optional: Generate lmdb dataset by `utils/lmdb_dataset.py`. If not, please overwrite `dataset:imagenet1k_lmdb` in yaml to `dataset:imagenet1k`.
    - The directory structure of `$DATA_ROOT` should look like this:
    ```
    ${DATA_ROOT}
    ├── imagenet
    └── imagenet_lmdb
    ```
    - Link the data:
    ```shell script
    ln -s YOUR_LMDB_DIR data/imagenet_lmdb
    ```

2. Cityscapes
    - Download data from [Cityscapes](https://www.cityscapes-dataset.com/).
    - unzip gtFine_trainvaltest.zip leftImg8bit_trainvaltest.zip
    - Link the data:
    ```shell script
    ln -s YOUR_DATA_DIR data/cityscapes
    ```
    - preprocess the data:
    ```shell script
    python3 tools/convert_cityscapes.py data/cityscapes --nproc 8
    ```

3. ADE20K
    - Download data from [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/).
    - unzip ADEChallengeData2016.zip
    - Link the data:
    ```shell script
    ln -s YOUR_DATA_DIR data/ade20k
    ```

4. COCO keypoints
    - Download data from [COCO](https://cocodataset.org/#download).
    - build tools
    ```shell script
    git clone https://github.com/cocodataset/cocoapi.git
    cd cocoapi/PythonAPI
    python3 setup.py build_ext --inplace
    python3 setup.py build_ext install
    make  # for nms
    ```
    - Unzip and Link the data:
    ```shell script
    ln -s YOUR_DATA_DIR data/coco
    ```
    - We also provide person detection result of COCO val2017 and test-dev2017 to reproduce our multi-person pose estimation results. Please download from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blWzzDXoz5BeFl8sWM-) or [GoogleDrive](https://drive.google.com/drive/folders/1fRUDNUDxe9fjqcRZ2bnF_TKMlO0nB_dk?usp=sharing).
    - Download and extract them under ```data/coco/person_detection_results```, and make them look like this:
    ```
    ${POSE_ROOT}
    |-- data
    `-- |-- coco
        `-- |-- annotations
            |   |-- person_keypoints_train2017.json
            |   `-- person_keypoints_val2017.json
            |-- person_detection_results
            |   |-- COCO_val2017_detections_AP_H_56_person.json
            |   |-- COCO_test-dev2017_detections_AP_H_609_person.json
            `-- images
                |-- train2017
                |   |-- 000000000009.jpg
                |   |-- 000000000025.jpg
                |   |-- 000000000030.jpg
                |   |-- ...
                `-- val2017
                    |-- 000000000139.jpg
                    |-- 000000000285.jpg
                    |-- 000000000632.jpg
                    |-- ...
    ```

## Miscellaneous
1. Plot keypoint detection results.
    ```shell script
    python3 tools/plot_coco.py --prediction output/results/keypoints_val2017_results_0.json --save-path output/vis/
    ```

2. About YAML config
- The codebase is a general ImageNet training framework using yaml config with several extension under `apps` dir, based on PyTorch.
    - YAML config with additional features
        - `${ENV}` in yaml config.
        - `_include` for hierachy config.
        - `_default` key for overwriting.
        - `xxx.yyy.zzz` for partial overwriting.
    - `--{{opt}} {{new_val}}` for command line overwriting.

3. If you find our work useful in your research please consider citing our paper:
    ```
    TBD
    ```

## Acknowledgement

* Code is inspired by [HR-NAS](https://github.com/dingmyu/HR-NAS)
* Work was done during Haoran's internship at [Baidu Research at USA](http://research.baidu.com/)
