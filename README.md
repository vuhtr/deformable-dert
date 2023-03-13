# Deformable DETR Working Repo
This repo makes slight changes to the original Deformable-DETR repo for easy training/finetuning purposes. And also addresses some errors. 

## Installation

```bash
pip install -r requirements.txt
cd ./models/ops
sh ./make.sh
python test.py
cd ../../
```

## Training

Structure of dataset:

```
    |-- custom_data/
        |-- annotations/
            |-- custom_train.json
            |-- custom_val.json
        |-- train2017
            |-- *.jpg
        |-- val2017
            |-- *.jpg
```

Download the pretrained-weights from the [original repo](https://github.com/fundamentalvision/Deformable-DETR#main-results) and put it in the current folder.

```bash
python -u main.py \
    --output_dir exps/exp0 \
    --with_box_refine --two_stage \
    --resume ./r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage-checkpoint.pth \
    --coco_path ./custom_data \
    --num_classes 8 \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-4 \
    --lr_drop 20
```

## Inference

```bash
python inference.py \
    --resume exps/exp0/checkpoint.pth \
    --input_dir ./inferences/test_images \
    --output_dir ./inferences/test_images_results \
    --num_classes 8
```