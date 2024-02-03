# PEMAE

## Datasets

Refer [PanCollection](https://github.com/XiaoXiao-Woo/PanCollection) for training and testing.

## Training

Refer the given args in [main_pretrain.py](./main_pretrain.py) for running the training script.

Below is an example:

```bash
python main_pretrain.py --model mae_vit_small \
--ms_chans 8 --rgb_c 4,2,1 \
--train_data_path /root/autodl-tmp/wv3/training_wv3/train_wv3.h5 \
--valid_data_path /root/autodl-tmp/wv3/training_wv3/valid_wv3.h5 \
--sensor wv3 --device cuda:0 \
--loss_type l1 --pos_type 2d_sincos \
--inp_type norm --attn_type sparse 
```

## Testing

Refer the given args in [test.py](./test.py) for running the testing script. You need to manually modify the variables to evaluate different models and datasets.

Refer the folder `qb_mae_vit_base_e4_l1_2d_sincos_norm_sparse` and `wv3_mae_vit_base_e4_l1_2d_sincos_norm_sparse` to access pre-trained weights using `PEMAE-B` with `4` ensembling counts on QuickBird and WorldView-3 datasets.

* Note: Due to the random masking strategy, the testing results of the given pre-trained weights may be slightly different from the reported results in the paper


## Acknowledgments

We are benefiting a lot from the following projects:

- [mae](https://github.com/facebookresearch/mae)
