import argparse
import time
import torch
from tqdm import tqdm
import torchmetrics.functional as MF
import wandb
from util.datasets import *
import util.metrics as mtc
from util.metrics_ori import *
import models_mae

def test(epoch, net, test_loader, criterion=torch.nn.L1Loss(), args=None):
    eval_metrics = ['MAE', 'CC', 'SAM', 'RMSE', 'ERGAS', 'PSNR', 'SSIM', 'RASE', 
                    'UQI', 'D_lambda', 'D_s', 'QNR', 'FCC', 'SF', 'SD', "Time"]
    tmp_results = {}
    net.eval()
    for metric in eval_metrics:
        tmp_results.setdefault(metric, [])
    device = next(net.parameters()).device
    with torch.no_grad():
        for data in tqdm(test_loader, 0):
            MS_image, PAN_image, hp_pan, hp_ms, up_ms, reference = data

            # Inputs and references...
            MS_image = MS_image.float().to(device)
            PAN_image = PAN_image.float().to(device)
            reference = reference.float().to(device)
            hp_pan = hp_pan.float().to(device)
            hp_ms = hp_ms.float().to(device)
            up_ms = up_ms.float().to(device)

            # Taking model output
            t_s = time.time()
            if args.inp_type == "hp":
                pred, *_ = net(hp_ms, hp_pan, up_ms)
            elif args.inp_type == "norm":
                pred, *_ = net(MS_image, PAN_image, up_ms)
            t_e = time.time()
            t= (t_e - t_s)
            
            up_ms[up_ms < 0] = 0.0
            up_ms[up_ms > 1.0] = 1.0
            MS_image[MS_image < 0] = 0.0
            MS_image[MS_image > 1.0] = 1.0
            reference[reference < 0] = 0.0
            reference[reference > 1.0] = 1.0

            pred[pred < 0] = 0.0
            pred[pred > 1.0] = 1.0
            data_range = (0., 1.)
            outputs = pred
            tmp_results['MAE'].append(criterion(outputs, reference).item())
            tmp_results['Time'].append(t)
            tmp_results['SSIM'].append(MF.structural_similarity_index_measure(outputs, reference, data_range=data_range).item())
            tmp_results['RMSE'].append(MF.root_mean_squared_error_using_sliding_window(outputs, reference).item())
            tmp_results['UQI'].append(MF.universal_image_quality_index(outputs, reference).item())
            tmp_results['ERGAS'].append(MF.error_relative_global_dimensionless_synthesis(outputs, reference).item())
            
            sam = MF.spectral_angle_mapper(outputs, reference)
            tmp_results['SAM'].append(sam.item() if torch.isfinite(sam) else 1)
            tmp_results['RASE'].append(MF.relative_average_spectral_error(outputs, reference).item())
            tmp_results['PSNR'].append(MF.peak_signal_noise_ratio(outputs, reference, data_range=data_range).item())
            tmp_results['CC'].append(cross_correlation(outputs, reference))
            
    eval_results = {"method": "EPMAE"}
    for metric in eval_metrics:
        mean = np.mean(tmp_results[metric])
        std = np.std(tmp_results[metric])
        eval_results[f'{metric}_mean'] = round(mean, 10)
        eval_results[f'{metric}_std'] = round(std, 10)
        
    filtered_dict = {k: v for k, v in eval_results.items() if type(v) == np.float64 and np.isnan(v) == False }
    print(filtered_dict)
    filtered_dict["epoch"] = epoch
    if not args.disable_wandb:
        wandb.log({"test": filtered_dict})
    
    return eval_results


def test_full(net, test_loader, hp=False):
    eval_metrics = ['MAE', 'CC', 'SAM', 'RMSE', 'ERGAS', 'PSNR', 'SSIM', 'RASE', 
                    'UQI', 'D_lambda', 'D_s', 'QNR', 'FCC', 'SF', 'SD', "Time"]
    tmp_results = {}
    net.eval()
    for metric in eval_metrics:
        tmp_results.setdefault(metric, [])
    device = next(net.parameters()).device
    with torch.no_grad():
        for data in tqdm(test_loader, 0):
            MS_image, PAN_image, hp_pan, hp_ms, up_ms = data

            # Inputs and references...
            MS_image = MS_image.float().to(device)
            PAN_image = PAN_image.float().to(device)
            hp_pan = hp_pan.float().to(device)
            hp_ms = hp_ms.float().to(device)
            up_ms = up_ms.float().to(device)

            # Taking model output
            t_s = time.time()
            if hp:
                pred, *_ = net(hp_ms, hp_pan, up_ms)
            else:
                pred, *_ = net(MS_image, PAN_image, up_ms)
            t_e = time.time()
            t= (t_e - t_s)
            
            up_ms[up_ms < 0] = 0.0
            up_ms[up_ms > 1.0] = 1.0
            pred[pred < 0] = 0.0
            pred[pred > 1.0] = 1.0
            tmp_results['Time'].append(t)
            d_lambda = mtc.D_lambda_torch(MS_image, pred)
            d_s = mtc.D_s_torch(MS_image, PAN_image, pred)
            qnr = (1 - d_lambda) * (1 - d_s)
            tmp_results['D_lambda'].append(d_lambda.item())
            tmp_results['D_s'].append(d_s.item())
            tmp_results['QNR'].append(qnr.item())
            
    eval_results = {"method": "EPMAE"}
    for metric in eval_metrics:
        mean = np.mean(tmp_results[metric])
        std = np.std(tmp_results[metric])
        eval_results[f'{metric}_mean'] = round(mean, 10)
        eval_results[f'{metric}_std'] = round(std, 10)

    filtered_dict = {k: v for k, v in eval_results.items() if type(v) == np.float64 and np.isnan(v) == False }
    print(filtered_dict)

    return eval_results



if __name__ == "__main__":

    # sensor = "qb"
    sensor = "wv3"
    # ms_chans = 4
    ms_chans = 8
    data_path = "/root/autodl-tmp/wv2/full_examples/test_wv2_OrigScale_multiExm1.h5"
    # data_path = "/root/autodl-tmp/wv3/full_examples/test_wv3_OrigScale_multiExm1.h5"
    # data_path = "/root/autodl-tmp/qb/full_examples/test_qb_OrigScale_multiExm1.h5"
    # data_path = "/root/autodl-tmp/wv2/reduced_examples/test_wv2_multiExm1.h5"
    # data_path = "/root/autodl-tmp/qb/reduced_examples/test_qb_multiExm1.h5"
    # data_path = "/root/autodl-tmp/wv3/reduced_examples/test_wv3_multiExm1.h5"
    device = torch.device("cuda:0")
    ensemble = 4
    ep = 954
    loss_type = "l1"
    pos_type = "2d_sincos"
    # pos_type = "trainable"
    inp_type = "norm"
    hp = False
    # model = "mae_vit_tiny"
    # model = "mae_vit_small"
    model = "mae_vit_base"
    # attn_type = "naive"
    attn_type = "sparse"
    output_dir = f"{sensor}_{model}_e{ensemble}_{loss_type}_{pos_type}_{inp_type}_{attn_type}"
    print(output_dir)
    model = models_mae.__dict__[model](img_size=16, 
                                       ms_chans=ms_chans, 
                                       ensemble=ensemble,
                                       pos_type=pos_type,
                                       attn_type=attn_type,
                                       )
    model.load_state_dict(torch.load(f"{output_dir}/checkpoint-{ep}.pth", map_location="cpu")["model"], strict=False)
    # model.load_state_dict(torch.load(f"wv3_tiny_e4_nohp/checkpoint-{ep}.pth")["model"], strict=False)
    model.to(device).eval()
    
    datasets = OriTestDataset(data_path)
    # datasets = ValidDataset(data_path)
    data_loader_test = torch.utils.data.DataLoader(
        datasets,
        batch_size=1,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    
    eval_results = test_full(model, data_loader_test, hp=hp)
    # eval_results = test(ep, model, data_loader_test, args=argparse.Namespace(**{"inp_type": inp_type, "disable_wandb": True}))
    