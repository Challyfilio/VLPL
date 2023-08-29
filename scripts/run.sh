#!/bin/bash

cd ..
#Use: bash run.sh CoOp oxford_pets rn50_ep50 end 16 1 False std

# custom config
DATA=/workspace/pycharm_project/CoOpProject/data

TRAINER=$1
DATASET=$2
CFG=$3   # config file
CTP=$4   # class token position (end or middle)
NCTX=$5  # number of context tokens
SHOTS=$6 # number of shots (1, 2, 4, 8, 16)
CSC=$7   # class-specific context (False or True)
STH=$8   # something about train

for SEED in 1 2 3; do
  DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/${STH}/seed${SEED}
  if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Skip this job"
  else
    echo "Run this job and save the output to ${DIR}"
    python train.py \
      --root ${DATA} \
      --seed ${SEED} \
      --trainer ${TRAINER} \
      --dataset-config-file configs/datasets/${DATASET}.yaml \
      --config-file configs/trainers/CoOp/${CFG}.yaml \
      --output-dir ${DIR} \
      TRAINER.COOP.N_CTX ${NCTX} \
      TRAINER.COOP.CSC ${CSC} \
      TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
      DATASET.NUM_SHOTS ${SHOTS}
  fi
done
python parse_test_res.py output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/${STH}

#--config-file configs/${trainers}/${TRAINER}/${CFG}.yaml \
