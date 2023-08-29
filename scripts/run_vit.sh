#!/bin/bash

echo "* * * * * * * * * * * * * * * * * * * * *"
echo "Hello, what a nice day! You are so great!"
echo "* * * * * * * * * * * * * * * * * * * * *"

cd ..

#Use: bash run_vit.sh CoOpVPE oxford_pets vit_b32_ep50 end 16 1 False

# custom config
DATA=/workspace/pycharm_project/CoOpProject/data

TRAINER=$1
DATASET=$2
CFG=$3   # config file
CTP=$4   # class token position (end or middle)
NCTX=$5  # number of context tokens
SHOTS=$6 # number of shots (1, 2, 4, 8, 16)
CSC=$7   # class-specific context (False or True)

list='1 2 3 4 5'
for TOKENS in ${list}; do
  for SEED in 1 2 3; do
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/std_p=${TOKENS}/seed${SEED}
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
        Tokens_Num ${TOKENS} \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS}
    fi
  done
  echo "$"
  python parse_test_res.py output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/std_p=${TOKENS}
done

echo "* * * * * * * * * * * * * * * * * * * * * * * * "

#for TOKENS in ${list}; do
#  echo "$"
#  python parse_test_res.py output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/std_p=${TOKENS}
#done
#--config-file configs/${trainers}/${TRAINER}/${CFG}.yaml \
