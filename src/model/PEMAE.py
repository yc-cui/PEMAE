import time
import os
import torch
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
import torchmetrics.functional.image as MF
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import pandas as pd
from torch.optim.lr_scheduler import StepLR
from sorcery import dict_of
from src.metric.cross_correlation import cross_correlation
from src.model.modules import check_and_make, regularize_inputs
from src.model.network import PEMAE


class PEMAEModel(pl.LightningModule):
    def __init__(self,
                 lr,
                 bands,
                 rgb_c,
                 sensor,
                 scatter,
                 pan,
                 ms,
                 focus,
                 dim,
                 depth,
                 ):
        super().__init__()

        self.automatic_optimization = False
        self.rgb_c = rgb_c
        self.sensor = sensor
        self.train_sensor = sensor
        self.bands = bands
        self.model = PEMAE(
            ensemble=scatter,
            depth=depth,
            embed_dim=dim,
            focus=focus,
            ms=ms,
            pan=pan,
        )

        self.reset_metrics()
        self.save_hyperparameters()
        self.visual_idx = [i for i in range(20)]

    def forward(self, masked_ms, masked_pan, pan, up_ms):
        pred = self.model(masked_ms, masked_pan, pan, up_ms)
        out = dict_of(pred)
        return out

    def loss(self, pred, gt, split="train"):
        loss = F.l1_loss(pred, gt)
        log_dict = {
            f"{split}/l1_loss": loss
        }
        return loss, log_dict

    def configure_optimizers(self):
        lr = self.hparams.lr
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        sche_opt = StepLR(opt, step_size=100, gamma=0.5)

        return [opt], [sche_opt]

    def training_step(self, batch, batch_idx):
        masked_ms, masked_pan, pan, up_ms, gt = batch["masked_ms"], batch["masked_pan"], batch["pan"], batch["up_ms"], batch["gt"]
        out = self.forward(masked_ms, masked_pan, pan, up_ms)
        pred = out["pred"]
        opt = self.optimizers()

        opt.zero_grad()
        total_loss, log_dict = self.loss(gt, pred)
        self.manual_backward(total_loss)
        opt.step()

        log_dict["lr"] = opt.param_groups[0]["lr"]
        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True)

    def on_train_epoch_end(self):
        sche_pf = self.lr_schedulers()
        sche_pf.step()

    def validation_step(self, batch, batch_idx):
        masked_ms, masked_pan, pan, up_ms, gt = batch["masked_ms"], batch["masked_pan"], batch["pan"], batch["up_ms"], batch["gt"]
        out = self.forward(masked_ms, masked_pan, pan, up_ms)
        pred = out["pred"]
        pred, gt, up_ms = regularize_inputs(pred, gt, up_ms)
        self.save_full_ref(pred, gt)
        if batch_idx in self.visual_idx:
            channel_indices = torch.tensor(self.rgb_c, device=self.device)
            up_ms_rgb = torch.index_select(up_ms, 1, channel_indices)
            gt_rgb = torch.index_select(gt, 1, channel_indices)
            pred_rgb = torch.index_select(pred, 1, channel_indices)
            err_rgb = torch.abs(pred - gt).mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
            err_rgb /= torch.max(err_rgb)
            rgb_imgs = torch.cat([
                up_ms_rgb,
                pred_rgb,
                gt_rgb,
                err_rgb], dim=0)

            if self.visual is None:
                self.visual = rgb_imgs
            else:
                self.visual = torch.cat([self.visual, rgb_imgs], dim=0)

    def on_validation_epoch_end(self):
        model_name = self.__class__.__name__
        eval_results = {"method": model_name}
        for metric in self.eval_metrics:
            mean = np.mean(self.metrics_all[metric])
            std = np.std(self.metrics_all[metric])
            eval_results[f'{metric}_mean'] = round(mean, 10)
            eval_results[f'{metric}_std'] = round(std, 10)
        filtered_dict = {k: v for k, v in eval_results.items() if isinstance(v, np.float64) and np.isnan(v) == False}
        self.log_dict(filtered_dict)
        filtered_dict["epoch"] = self.current_epoch
        csv_path = os.path.join(self.logger.save_dir, "metrics.csv")
        pd.DataFrame.from_dict(
            [filtered_dict]).to_csv(
            csv_path,
            mode="a",
            index=False,
            header=False if os.path.exists(csv_path) else True)

        grid = make_grid(self.visual, nrow=4, padding=2, normalize=False, scale_each=False, pad_value=0)
        image_grid = grid.permute(1, 2, 0).cpu().numpy()
        check_and_make(f"visual-{model_name}")
        save_path = f"visual-{model_name}/{self.current_epoch}.jpg"
        plt.imsave(save_path, image_grid)
        # self.logger.log_image(key="visual", images=[save_path])
        self.reset_metrics()

    def test_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0:  # reduced
            masked_ms, masked_pan, pan, up_ms, gt = batch["masked_ms"], batch["masked_pan"], batch["pan"], batch["up_ms"], batch["gt"]
            out = self.forward(masked_ms, masked_pan, pan, up_ms)
            pred = out["pred"]
            pred, gt, up_ms = regularize_inputs(pred, gt, up_ms)
            self.save_full_ref(pred, gt, "test")
        else:  # full
            masked_ms, masked_pan, pan, up_ms, gt = batch["masked_ms"], batch["masked_pan"], batch["pan"], batch["up_ms"], batch["gt"]
            t_s = time.time()
            out = self.forward(masked_ms, masked_pan, pan, up_ms)
            pred = out["pred"]
            t_e = time.time()
            pred, up_ms = regularize_inputs(pred, up_ms)
            self.save_no_ref(masked_ms, pan, pred, "test")
            self.record_metrics('Time', torch.Tensor([t_e - t_s]), "test")

    def on_test_epoch_start(self):
        self.reset_metrics("test")

    def on_test_epoch_end(self):
        model_name = self.__class__.__name__
        eval_results = {"method": model_name}
        for metric in self.eval_metrics:
            mean = np.mean(self.metrics_all[metric])
            std = np.std(self.metrics_all[metric])
            eval_results[f'{metric}_mean'] = round(mean, 10)
            eval_results[f'{metric}_std'] = round(std, 10)
        filtered_dict = {k: v for k, v in eval_results.items() if isinstance(
            v, np.float64) and np.isnan(v) == False and "std" not in k}
        print(filtered_dict)
        filtered_dict["epoch"] = "testing"
        csv_path = os.path.join(self.logger.save_dir, f"{self.sensor}.csv")
        pd.DataFrame.from_dict(
            [filtered_dict]).to_csv(
            csv_path,
            mode="a",
            index=False,
            header=False if os.path.exists(csv_path) else True)
        self.reset_metrics()

    def save_full_ref(self, pred, gt, split="val"):
        data_range = (0., 1.)
        self.record_metrics('MAE', F.l1_loss(pred, gt), split)
        self.record_metrics('SSIM', MF.structural_similarity_index_measure(pred, gt, data_range=data_range), split)
        self.record_metrics('RMSE', MF.root_mean_squared_error_using_sliding_window(pred, gt), split)
        self.record_metrics('ERGAS', MF.error_relative_global_dimensionless_synthesis(pred, gt) / 16., split)
        self.record_metrics('SAM', MF.spectral_angle_mapper(pred, gt), split)
        self.record_metrics('RASE', MF.relative_average_spectral_error(pred, gt), split)
        self.record_metrics('PSNR', MF.peak_signal_noise_ratio(pred, gt, data_range=data_range), split)
        self.record_metrics('UQI', MF.universal_image_quality_index(pred, gt), split)
        self.record_metrics('CC', cross_correlation(pred, gt), split)

    def save_no_ref(self, lrms, pan, pred, split="val"):
        d_lambda = MF.spectral_distortion_index(lrms, pred)
        d_s = MF.spatial_distortion_index(pred, lrms, pan.repeat(1, lrms.shape[1], 1, 1))
        qnr = (1 - d_lambda) * (1 - d_s)
        self.record_metrics('D_lambda', d_lambda, split)
        self.record_metrics('D_s', d_s, split)
        self.record_metrics('QNR', qnr, split)

    def reset_metrics(self, split="val"):
        self.eval_metrics = ['MAE', 'SAM', 'RMSE', 'ERGAS', 'PSNR', 'SSIM', 'RASE', 'UQI',
                             'D_lambda', 'D_s', 'QNR', 'FCC', 'SF', 'SD', "Time", "CC",]
        self.eval_metrics = [f"{split}/" + i for i in self.eval_metrics]
        tmp_results = {}
        for metric in self.eval_metrics:
            tmp_results.setdefault(metric, [])

        self.metrics_all = tmp_results
        self.visual = None

    def record_metrics(self, k, v, split="val"):
        if torch.isfinite(v):
            self.metrics_all[f'{split}/' + k].append(v.item())
