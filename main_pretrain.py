import argparse
import datetime
import json
import os
import time
import numpy as np
import torch
import wandb
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import timm.optim.optim_factory as optim_factory
from util.datasets import TrainDataset, ValidDataset
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import models_mae
from engine_pretrain import train_one_epoch
from test import test


def get_args_parser():
    parser = argparse.ArgumentParser('PEMAE training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--input_size', default=16, type=int)
    parser.add_argument('--sensor', default="wv3", type=str)
    parser.add_argument('--ms_chans', default=8, type=int)
    parser.add_argument('--ensemble', default=4, type=int)
    parser.add_argument('--rgb_c', default='2,1,0')
    parser.add_argument('--train_data_path', default="/root/autodl-tmp/wv3/training_wv3/train_wv3.h5", type=str)
    parser.add_argument('--valid_data_path', default='/root/autodl-tmp/wv3/training_wv3/valid_wv3.h5', type=str)
    parser.add_argument('--test_freq', default=1, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--warmup_epochs', type=int, default=40)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--disable_wandb', action='store_true')
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--model', default='mae_vit_tiny', choices=["mae_vit_tiny", "mae_vit_small", "mae_vit_base", "mae_vit_cust"])
    parser.add_argument('--decoder_embed_dim', type=int)
    parser.add_argument('--decoder_depth', type=int)
    parser.add_argument('--loss_type', default="l1", choices=["l1", "l2"])
    parser.add_argument('--pos_type', default="2d_sincos", choices=["2d_sincos", "trainable"])
    parser.add_argument('--inp_type', default="hp", choices=["hp", "norm"])
    parser.add_argument('--attn_type', default="sparse", choices=["sparse", "naive"])

    return parser


def main(args):
    
    # define logs
    if args.model != "mae_vit_cust":
        output_dir = f"{args.sensor}_{args.model}_e{args.ensemble}_{args.loss_type}_{args.pos_type}_{args.inp_type}_{args.attn_type}"
    else:
        output_dir = f"{args.sensor}_{args.model}_dim{args.decoder_embed_dim}_depth{args.decoder_depth}"

    log_writer = SummaryWriter(log_dir=output_dir)
    
    rgbc = [int(c) for c in args.rgb_c.split(",")]
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not args.disable_wandb:
        # os.environ['WANDB_MODE'] = 'offline'
        wandb.init(project="PEMAE", tags=["PEMAE"], name=output_dir)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    # fix the seed for reproducibility
    device = torch.device(args.device)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # define datasets
    dataset_train = TrainDataset(args.train_data_path)
    dataset_valid = ValidDataset(args.valid_data_path)
    data_loader_train = DataLoader(dataset_train,
                                   batch_size=args.batch_size,
                                   num_workers=args.num_workers,
                                   pin_memory=args.pin_mem,
                                   drop_last=False,
                                   shuffle=True,
                                   )
    data_loader_valid = DataLoader(dataset_valid,
                                  batch_size=1,
                                  num_workers=args.num_workers,
                                  pin_memory=args.pin_mem,
                                  drop_last=False,
                                  )
    
    # define model
    if args.model == "mae_vit_cust":
        model = models_mae.__dict__[args.model](img_size=args.input_size, 
                                                ms_chans=args.ms_chans,
                                                ensemble=args.ensemble,
                                                pos_type=args.pos_type,
                                                attn_type=args.attn_type,
                                                decoder_embed_dim=args.decoder_embed_dim,
                                                decoder_depth=args.decoder_depth
                                                )
    else:
        model = models_mae.__dict__[args.model](img_size=args.input_size, 
                                                ms_chans=args.ms_chans,
                                                ensemble=args.ensemble,
                                                pos_type=args.pos_type,
                                                attn_type=args.attn_type,
                                                )
            

    model.to(device)
    # model = torch.compile(model, mode="reduce-overhead")
    param_groups = optim_factory.param_groups_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()
    
    # training
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        
        train_stats = train_one_epoch(model, 
                                      data_loader_train,
                                      optimizer, 
                                      device, 
                                      epoch, 
                                      loss_scaler,
                                      log_writer, 
                                      args,
                                      )
        
        if (epoch % args.test_freq == 0 or epoch + 1 == args.epochs):
            if epoch > 1:
                misc.save_model(output_dir,
                                epoch,
                                model,
                                optimizer,
                                loss_scaler,
                                )
        
            test(epoch, 
                 model, 
                 data_loader_valid, 
                 args=args
                 )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}

        if log_writer is not None:
            log_writer.flush()
        with open(os.path.join(output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
    
#  (?!.*std)^.*$