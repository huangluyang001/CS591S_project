# CS591S_project

## Reinforce-guided dialogue generation

### usage

1. download data from https://github.com/facebookresearch/ParlAI

2. run preprocess and make vocab

```
python preprocess/prepare_data.py
```

```
python make_vocab.py
```
Since this step takes some time (tokenization), we provide processed data on: https://drive.google.com/drive/folders/1EWWrptQ5uIMB7skVNvdcGNKzYi_sYOBJ?usp=sharing

3. train your ML model

```
export DATA=[YOUR/PATH/TO/DATA]
```

```
python train_abstractor.py --path [YOUR/ABS/MODEL/PATH]
```

4. train RL model based on best ML model
```
python train_abstractor_rl.py --path [YOUR/RL/MODEL/PATH] --abs_dir [YOUR/ABS/MODEL/PATH] --bleu(--f1)
```

We provide our trained model here: https://drive.google.com/drive/folders/1EWWrptQ5uIMB7skVNvdcGNKzYi_sYOBJ?usp=sharing

5. decode
```
python decode_abs.py --path [YOUR/RESULT/PATH] --avs_dir [YOUR/RL/MODEL/PATH] --reverse --test
```

We provide our decoded results here: https://drive.google.com/drive/folders/1EWWrptQ5uIMB7skVNvdcGNKzYi_sYOBJ?usp=sharing

6. evaluate

```
python eval_dialog.py --out_path [YOUR/OUT/PATH] --ref_path [YOUR/PATH/TO/DATA] --type pg
```
