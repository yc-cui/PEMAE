import math
import sys
from typing import Iterable
import torch
import wandb 
import torch.nn.functional as F
import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, 
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, 
                    epoch: int, 
                    loss_scaler,
                    log_writer=None,
                    args=None,
                    ):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    
    if args.loss_type == "l1":
        loss_fn = F.l1_loss
    elif args.loss_type == "l2":
        loss_fn = F.mse_loss
        
    optimizer.zero_grad()
    accum_iter = args.accum_iter
    for data_iter_step, (ms, pan, hp_pan, hp_ms, up_ms, gt) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)     

        gt = gt.float().to(device, non_blocking=True)
        up_ms = up_ms.float().to(device, non_blocking=True)

        if args.inp_type == "hp":
            inp_pan = hp_pan.float().to(device, non_blocking=True)    
            inp_ms = hp_ms.float().to(device, non_blocking=True)      
        elif args.inp_type == "norm":
            inp_pan = pan.float().to(device, non_blocking=True)    
            inp_ms = ms.float().to(device, non_blocking=True)    

        with torch.cuda.amp.autocast():
            pred, *_ = model(inp_ms, inp_pan, up_ms)
            loss = loss_fn(gt, pred)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)     
        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()       
        torch.cuda.synchronize()        
        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)     
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)        

        if not args.disable_wandb:
            wandb.log({"train": {"recon_l1_loss": loss.item(),  "lr": optimizer.param_groups[0]["lr"]}})

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}