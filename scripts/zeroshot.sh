#!/bin/bash

cd ..

# custom config
DATA=/workspace/pycharm_project/CoOpProject/data
TRAINER=ZeroshotCLIP
CFG=$1 # rn50, rn101, vit_b32 or vit_b16

list_datasets='oxford_pets oxford_flowers fgvc_aircraft stanford_cars dtd eurosat food101 sun397 ucf101 caltech101'

for DS in ${list_datasets}; do
  python train.py \
    --root ${DATA} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DS}.yaml \
    --config-file configs/trainers/CoOp/${CFG}.yaml \
    --output-dir output/${TRAINER}/${CFG}/${DS} \
    --eval-only
done
