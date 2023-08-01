## How to Run

We provide the running scripts in `scripts/`.

Make sure you change the path in `DATA` and run the commands under the main directory `VLPL/`.

### Few-Shot Learning

All you need is `VLPL/scripts/test.sh`, which contains seven input arguments.

```
TRAINER=$1
CFG=$2   # config file
CTP=$3   # class token position (end or middle)
NCTX=$4  # number of context tokens
SHOTS=$5 # number of shots (1, 2, 4, 8, 16)
CSC=$6   # class-specific context (False or True)
STH=$7   # something about train
```

`list_datasets` takes as input a dataset name, like `imagenet` or `caltech101`. The valid names are the files' names in `CoOp/configs/datasets/`.

Use: 
```
bash test.sh [model] [config_file] [cls_pos] [prompt_length] [n-shot] [csc] [sth]
```

Example:
```
bash test.sh CoOpVPE vit_b32_ep50 end 16 1 False std
```

After the experiments are finished, you can use `parse_test_res.py` to calculate the average results instead of manually looking into the log files. Say the structure of `output/` is

```
output
|–– caltech101/
|   |–– CoOp/
|   |   |–– rn50_16shots/
|   |   |   |–– nctx16_cscFalse_ctpend/
|   |   |   |   |–– seed1/
|   |   |   |   |–– seed2/
|   |   |   |   |–– seed3/
|   |   |–– rn50_8shots/
|   |   |   |–– nctx16_cscFalse_ctpend/
|   |   |   |   |–– seed1/
|   |   |   |   |–– seed2/
|   |   |   |   |–– seed3/
```

To calculate the average results for the folder `rn50_16shots/nctx16_cscFalse_ctpend/`, you can run

```bash
python parse_test_res.py output/caltech101/CoOp/rn50_16shots/nctx16_cscFalse_ctpend
```

Then, you will see something like this in your terminal

```bash
Parsing files in output/caltech101/CoOp/rn50_16shots/nctx16_cscFalse_ctpend
file: output/caltech101/CoOp/rn50_16shots/nctx16_cscFalse_ctpend/seed1/log.txt. accuracy: 91.81%. error: 8.19%.
file: output/caltech101/CoOp/rn50_16shots/nctx16_cscFalse_ctpend/seed2/log.txt. accuracy: 92.01%. error: 7.99%.
file: output/caltech101/CoOp/rn50_16shots/nctx16_cscFalse_ctpend/seed3/log.txt. accuracy: 92.17%. error: 7.83%.
===
Summary of directory: output/caltech101/CoOp/rn50_16shots/nctx16_cscFalse_ctpend
* accuracy: 92.00% +- 0.15%
* error: 8.00% +- 0.15%
===
```

**How to initialize the context tokens with pre-trained word vectors?** Specify the words for the parameter `TRAINER.COOP.CTX_INIT` in your config file. In our paper, we use `configs/trainers/rn50_ctxv1.yaml` (give this file to `--config-file`, see `scripts/coop/main.sh`), which uses "a photo of a" as the initialization words.

**How to visualize nearest words for the learned context tokens?** All you need is `interpret_prompt.py`. Say the learned tokens are saved in `a/b/c/prompt_learner/model.pth.tar` and you would like to see the top-3 nearest words for each token. In this case, run `python interpret_prompt.py a/b/c/prompt_learner/model.pth.tar 3`

```bash
# Don't need to use rn5_ep50 here as no training is performed
bash scripts/coop/eval.sh imagenetv2 rn50
```

The default setting is `SHOTS=16`. Feel free to modify the script.

Again, you can use `parse_test_res.py` to automate the calculation of average performance. This time you should append `--test-log`, e.g., `python parse_test_res.py directory --test-log`.

### Zero-Shot CLIP

See `VLPL/scripts/zeroshot.sh`.

### Linear Probe CLIP

Please move to [lpclip/](lpclip/).