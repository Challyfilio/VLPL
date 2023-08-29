#!/bin/bash

echo "* * * * * * * * * * * * * * * * * * * * *"
echo "Hello, what a nice day! You are so great!"
echo "* * * * * * * * * * * * * * * * * * * * *"

cd ..

#Use: bash test.sh [model] [config_file] [cls_pos] [prompt_length] [n-shot] [csc] [sth]
#Use: bash test.sh CoOpVPE vit_b32_ep50 end 16 1 False std

# custom config
DATA=/workspace/pycharm_project/CoOpProject/data

TRAINER=$1
CFG=$2   # config file
CTP=$3   # class token position (end or middle)
NCTX=$4  # number of context tokens
SHOTS=$5 # number of shots (1, 2, 4, 8, 16)
CSC=$6   # class-specific context (False or True)
STH=$7   # something about train

list='-1'
#'oxford_pets oxford_flowers fgvc_aircraft stanford_cars dtd eurosat food101 sun397 ucf101 caltech101 imagenet'
#'oxford_pets fgvc_aircraft eurosat ucf101 caltech101'
list_datasets='oxford_pets oxford_flowers fgvc_aircraft stanford_cars dtd eurosat food101 sun397 ucf101 caltech101 imagenet'

for DS in ${list_datasets}; do
  for SEED in 1; do
    DIR=output/${DS}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/${STH}/seed${SEED}
    #    DIR=output/${DS}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/std_p=${list}/seed${SEED}
    if [ -d "$DIR" ]; then
      echo "Results are available in ${DIR}. Skip this job"
    else
      echo "Run this job and save the output to ${DIR}"
      python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DS}.yaml \
        --config-file configs/trainers/CoOp/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS}
    fi
  done
  python parse_test_res.py output/${DS}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/${STH}
  #  python parse_test_res.py output/${DS}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/std_p=${list}
done

#for TOKENS in ${list}; do
#  echo "$"
#  python parse_test_res.py output/${DS}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/std_p=${TOKENS}
#done

#--config-file configs/${trainers}/${TRAINER}/${CFG}.yaml \
