
import argparse
import os

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger, CSVLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.dataset.NBUData import plNBUDataset
from src.model.PEMAE import PEMAEModel


sensor2dir = {
    'wv2': '~/autodl-tmp/5 WorldView-2',
    'gf1': '~/autodl-tmp/3 Gaofen-1',
    'ik': '~/autodl-tmp/1 IKONOS',
    'wv3': '~/autodl-tmp/6 WorldView-3',
    'wv4': '~/autodl-tmp/4 WorldView-4',
    'qb': '~/autodl-tmp/2 QuickBird',
}


def get_args_parser():
    parser = argparse.ArgumentParser('PEMAE training', add_help=False)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--epochs', default=2, type=int)
    parser.add_argument('--ms_chans', default=4, type=int, help="number of ms bands")
    parser.add_argument('--scatter_num', default=4, type=int, help="number of ensemble")
    parser.add_argument('--pan_down', action='store_true', help="use downsampled pan as input")
    parser.add_argument('--lrms_up', action='store_true', help="use upsampled ms at bottleneck")
    parser.add_argument('--focus', default=6, type=int, help="focus factor")
    parser.add_argument('--dim', default=32, type=int, help="embedding dimension")
    parser.add_argument('--depth', default=4, type=int, help="encoder and decoder depth")
    parser.add_argument('--rgb_c', default='2,1,0', help="RGB channel for visualization")
    parser.add_argument('--train_sensor', default='wv2', type=str, help="use which sensor to train")
    parser.add_argument('--test_sensors', default='ik,gf1,wv3,wv4,qb', type=str, help="use which sensors to test")
    parser.add_argument('--test_freq', default=1, type=int, help="frequency to evaluate the model")
    parser.add_argument('--device', default=0, type=int, help="training at which GPU device")
    parser.add_argument('--seed', default=42, type=int, help="random seed")
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help="learning rate")
    parser.add_argument('--wandb', action='store_true', help="whether to use wandb for logging")
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    return parser


def main(args):

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    model_name = PEMAEModel.__name__

    output_dir = f"log_m={model_name}_s={args.train_sensor}_sd={args.seed}_s={args.scatter_num}_pan={args.pan_down}_ms={args.lrms_up}_f={args.focus}_di={args.dim}_de={args.depth}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    seed_everything(args.seed)

    dir_train_sensor = sensor2dir[args.train_sensor]
    dataset = plNBUDataset(dir_train_sensor,
                           dir_train_sensor,
                           args.batch_size,
                           args.num_workers,
                           args.pin_mem,
                           args.seed,
                           )
    model = PEMAEModel(lr=args.lr,
                       bands=args.ms_chans,
                       rgb_c=[int(c) for c in args.rgb_c.split(",")],
                       sensor=args.train_sensor,
                       scatter=args.scatter_num,
                       pan=args.pan_down,
                       ms=args.lrms_up,
                       focus=args.focus,
                       dim=args.dim,
                       depth=args.depth,
                       )

    if args.wandb:
        wandb_logger = WandbLogger(project=model_name, name=output_dir, save_dir=output_dir)
    else:
        wandb_logger = [CSVLogger(name=output_dir, save_dir=output_dir)]
        wandb_logger.append(TensorBoardLogger(name=output_dir, save_dir=output_dir))

    model_checkpoint = ModelCheckpoint(dirpath=output_dir,
                                       monitor='val/PSNR_mean',
                                       mode="max",
                                       save_top_k=1,
                                       auto_insert_metric_name=False,
                                       filename='ep={epoch}_PSNR={val/PSNR_mean:.4f}',
                                       every_n_epochs=args.test_freq,
                                       save_last=True
                                       )

    trainer = pl.Trainer(max_epochs=args.epochs,
                         accelerator="gpu",
                         devices=[args.device],
                         logger=wandb_logger,
                         check_val_every_n_epoch=args.test_freq,
                         callbacks=[model_checkpoint],
                         )

    trainer.fit(model, dataset)
    trainer.test(ckpt_path="best", datamodule=dataset)

    sensors = args.test_sensors.split(",")
    dir_test_sensors = [sensor2dir[s] for s in sensors]
    for sensor, dir_test_sensor in zip(sensors, dir_test_sensors):
        dataset = plNBUDataset(dir_train_sensor,
                               dir_test_sensor,
                               args.batch_size,
                               args.num_workers,
                               args.pin_mem,
                               args.seed,
                               )
        model.sensor = sensor
        trainer.test(model, ckpt_path="best", datamodule=dataset)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
