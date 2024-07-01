import gc
import os
from PIL import Image
import numpy as np

import hydra
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import InterpolationMode
from os.path import join

from featup.datasets.JitteredImage import apply_jitter, sample_transform
from featup.datasets.util import get_dataset, SingleImageDataset
from featup.downsamplers import SimpleDownsampler, AttentionDownsampler
from featup.featurizers.util import get_featurizer
from featup.layers import ChannelNorm
from featup.losses import TVLoss, SampledCRFLoss, entropy
from featup.upsamplers import get_upsampler
from featup.util import pca, RollingAvg, unnorm, norm, prep_image

torch.multiprocessing.set_sharing_strategy('file_system')


class ScaleNet(torch.nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.net = torch.nn.Conv2d(dim, 1, 1)
        with torch.no_grad():
            self.net.weight.copy_(self.net.weight * .1)
            self.net.bias.copy_(self.net.bias * .1)

    def forward(self, x):
        return torch.exp(self.net(x) + .1).clamp_min(.0001)


class JBUFeatUp(pl.LightningModule):
    def __init__(self,
                 model_type,
                 activation_type,
                 n_jitters,
                 max_pad,
                 max_zoom,
                 kernel_size,
                 final_size,
                 lr,
                 random_projection,
                 predicted_uncertainty,
                 crf_weight,
                 filter_ent_weight,
                 tv_weight,
                 upsampler,
                 downsampler,
                 chkpt_dir,
                 ):
        super().__init__()
        self.model_type = model_type
        self.activation_type = activation_type
        self.n_jitters = n_jitters
        self.max_pad = max_pad
        self.max_zoom = max_zoom
        self.kernel_size = kernel_size
        self.final_size = final_size
        self.lr = lr
        self.random_projection = random_projection
        self.predicted_uncertainty = predicted_uncertainty
        self.crf_weight = crf_weight
        self.filter_ent_weight = filter_ent_weight
        self.tv_weight = tv_weight
        self.chkpt_dir = chkpt_dir

        self.model, self.patch_size, self.dim = get_featurizer(model_type, activation_type, num_classes=1000)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model = torch.nn.Sequential(self.model, ChannelNorm(self.dim))
        self.upsampler = get_upsampler(upsampler, self.dim)

        if downsampler == 'simple':
            self.downsampler = SimpleDownsampler(self.kernel_size, self.final_size)
        elif downsampler == 'attention':
            self.downsampler = AttentionDownsampler(self.dim, self.kernel_size, self.final_size, blur_attn=True)
        else:
            raise ValueError(f"Unknown downsampler {downsampler}")

        if self.predicted_uncertainty:
            self.scale_net = ScaleNet(self.dim)

        self.avg = RollingAvg(20)

        self.crf = SampledCRFLoss(
            alpha=.1,
            beta=.15,
            gamma=.005,
            w1=10.0,
            w2=3.0,
            shift=0.00,
            n_samples=1000)
        self.tv = TVLoss()

        self.automatic_optimization = False

    def forward(self, x):
        return self.upsampler(self.model(x))

    def project(self, feats, proj):
        if proj is None:
            return feats
        else:
            return torch.einsum("bchw,bcd->bdhw", feats, proj)

    def training_step(self, batch, batch_idx):
        return None

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            img = batch['img'] if isinstance(batch, dict) else batch[0]
            lr_feats = self.model(img)

            hr_feats = self.upsampler(lr_feats, img)

            if hr_feats.shape[2] != img.shape[2]:
                hr_feats = torch.nn.functional.interpolate(hr_feats, img.shape[2:], mode="bilinear")

            # Save the images and hr_feats
            img_dir = os.path.join(self.logger.save_dir, self.logger.name, 'imgs')
            os.makedirs(img_dir, exist_ok=True)

            # Save original image
            orig_img = img[0].cpu().numpy().transpose(1, 2, 0)
            orig_img = ((orig_img - orig_img.min()) / (orig_img.max() - orig_img.min()) * 255).astype(np.uint8)
            orig_img_path = os.path.join(img_dir, f"image_{batch_idx}.png")
            Image.fromarray(orig_img).save(orig_img_path)

            # Save high-resolution features image
            [red_hr_feats], fit_pca = pca([hr_feats[0].unsqueeze(0)])
            hr_feats_img = red_hr_feats[0].cpu().numpy().transpose(1, 2, 0)
            hr_feats_img = ((hr_feats_img - hr_feats_img.min()) / (hr_feats_img.max() - hr_feats_img.min()) * 255).astype(np.uint8)
            hr_feats_img_path = os.path.join(img_dir, f"hr_feats_image_{batch_idx}.png")
            Image.fromarray(hr_feats_img).save(hr_feats_img_path)

            # Save high-resolution features tensor
            #hr_feats_tensor_path = os.path.join(img_dir, f"hr_feats_tensor_{batch_idx}.pt")
            #torch.save(hr_feats, hr_feats_tensor_path)

            #writer = self.logger.experiment

            #writer.add_image(f"test_viz/hr_feats_image_{batch_idx}", red_hr_feats[0], self.global_step)

            #writer.flush()

    def configure_optimizers(self):
        all_params = []
        all_params.extend(list(self.downsampler.parameters()))
        all_params.extend(list(self.upsampler.parameters()))

        if self.predicted_uncertainty:
            all_params.extend(list(self.scale_net.parameters()))

        return torch.optim.NAdam(all_params, lr=self.lr)


@hydra.main(config_path="configs", config_name="jbu_upsampler.yaml", version_base=None)
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print(cfg.output_root)
    seed_everything(seed=0, workers=True)

    load_size = 224

    if cfg.model_type == "dinov2":
        final_size = 16
        kernel_size = 14
    elif cfg.model_type == "resnet50":
        final_size = 14
        kernel_size = 35
    else:
        final_size = 14
        kernel_size = 16

    name = (f"test_{cfg.model_type}_{cfg.upsampler_type}_"
            f"{cfg.dataset}_{cfg.downsampler_type}_"
            f"crf_{cfg.crf_weight}_tv_{cfg.tv_weight}"
            f"_ent_{cfg.filter_ent_weight}")

    log_dir = join(cfg.output_root, f"logs/jbu/{name}")
    chkpt_dir = join(cfg.output_root, f"checkpoints/jbu/{name}.ckpt")
    os.makedirs(log_dir, exist_ok=True)

    model = JBUFeatUp(
        model_type=cfg.model_type,
        activation_type=cfg.activation_type,
        n_jitters=cfg.n_jitters,
        max_pad=cfg.max_pad,
        max_zoom=cfg.max_zoom,
        kernel_size=kernel_size,
        final_size=final_size,
        lr=cfg.lr,
        random_projection=cfg.random_projection,
        predicted_uncertainty=cfg.outlier_detection,
        crf_weight=cfg.crf_weight,
        filter_ent_weight=cfg.filter_ent_weight,
        tv_weight=cfg.tv_weight,
        upsampler=cfg.upsampler_type,
        downsampler=cfg.downsampler_type,
        chkpt_dir=chkpt_dir
    )

    transform = T.Compose([
        T.Resize(load_size, InterpolationMode.BILINEAR),
        T.CenterCrop(load_size),
        T.ToTensor(),
        norm])

    dataset = get_dataset(
        cfg.pytorch_data_dir,
        cfg.dataset,
        "train",
        transform=transform,
        target_transform=None,
        include_labels=False)

    # Ensure to load different images
    # indices from 0 to 299
    indices = list(range(300))
    subset = Subset(dataset, indices)
    val_loader = DataLoader(
        subset, 1, shuffle=False, num_workers=cfg.num_workers)

    tb_logger = TensorBoardLogger(log_dir, default_hp_metric=False)
    callbacks = [ModelCheckpoint(chkpt_dir[:-5], every_n_epochs=1)]

    trainer = Trainer(
        accelerator='gpu',
        strategy=DDPStrategy(find_unused_parameters=True),
        devices=cfg.num_gpus,
        max_epochs=cfg.epochs,
        logger=tb_logger,
        val_check_interval=1,
        log_every_n_steps=1,
        callbacks=callbacks,
        reload_dataloaders_every_n_epochs=1,
    )

    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

    # Load the checkpoint
    checkpoint_path = "~/experiments/checkpoints/jbu/vit_jbu_stack_marr_attention_crf_0.001_tv_0.0_ent_0.0/epoch=0-step=15900.ckpt"
    model = JBUFeatUp.load_from_checkpoint(checkpoint_path, 
                                           model_type=cfg.model_type,
                                           activation_type=cfg.activation_type,
                                           n_jitters=cfg.n_jitters,
                                           max_pad=cfg.max_pad,
                                           max_zoom=cfg.max_zoom,
                                           kernel_size=kernel_size,
                                           final_size=final_size,
                                           lr=cfg.lr,
                                           random_projection=cfg.random_projection,
                                           predicted_uncertainty=cfg.outlier_detection,
                                           crf_weight=cfg.crf_weight,
                                           filter_ent_weight=cfg.filter_ent_weight,
                                           tv_weight=cfg.tv_weight,
                                           upsampler=cfg.upsampler_type,
                                           downsampler=cfg.downsampler_type,
                                           chkpt_dir=chkpt_dir
                                           )

    trainer.validate(model, val_loader)


if __name__ == "__main__":
    my_app()
