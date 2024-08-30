
import torch.nn as nn
import torch
from mmcv.ops import ModulatedDeformConv2dPack as Conv2d

from .modules import LayerNorm2d, Refine, PatchEmbed, Block


class PEMAE(nn.Module):
    """ Pixel-Wise Ensembled Masked Autoencoder
    """

    def __init__(self,
                 img_size=64,
                 ms_chans=4,
                 ensemble=4,
                 embed_dim=32,
                 down_factor=4,
                 depth=4,
                 num_heads=1,
                 mlp_ratio=2.,
                 norm_layer=LayerNorm2d,
                 focus=6,
                 ms=True,
                 pan=True,
                 ):
        super().__init__()
        self.img_size = img_size
        self.down_factor = down_factor
        self.embed_dim = embed_dim
        decoder_embed_dim = embed_dim
        self.focus = focus
        self.use_ms = ms
        self.use_pan = pan

        # patch embed
        self.patch_embed = PatchEmbed(img_size, ms_chans, embed_dim)
        self.pan_embed = PatchEmbed(img_size, ms_chans, embed_dim)
        self.pan_decoder_embed = PatchEmbed(img_size, ms_chans, decoder_embed_dim)
        self.upms_decoder_embed = PatchEmbed(img_size, ms_chans, decoder_embed_dim)
        self.pandown_decoder_embed = PatchEmbed(img_size, ms_chans, decoder_embed_dim)
        self.decoder_embed = Conv2d(embed_dim, decoder_embed_dim, kernel_size=1, stride=1, padding=0)
        self.decoder_pred = Conv2d(decoder_embed_dim, ms_chans, kernel_size=3, stride=1, padding=1)
        # 1 1 D'
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=False)

        # encoder and decoder
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, norm_layer=norm_layer, focus=self.focus)
            for _ in range(depth)])

        self.dec_blocks = nn.ModuleList([
            Block(decoder_embed_dim, num_heads, mlp_ratio, norm_layer=norm_layer, focus=self.focus)
            for _ in range(depth)])

        # ensembler
        self.scatter_num = ensemble
        kernel_num = ms_chans * ensemble
        self.ensembler = Refine(kernel_num, ms_chans)

    def get_global_pos(self, x):
        """ get scatter position
        """
        size = x.shape[-1]
        N, L, *_ = x.flatten(2).transpose(1, 2).shape
        down_factor = self.down_factor

        row = torch.arange(0, size * down_factor, down_factor, device=x.device).unsqueeze(-1)
        col = size * down_factor * down_factor * torch.arange(size, device=x.device)
        corner = (row + col).T  # top left
        offset = torch.randint(down_factor, (N, size, size, 2), device=x.device)  # offset to right and bottom
        sample_idxs = corner + offset[..., 0] * size * down_factor + offset[..., 1]
        global_sampled_pos = sample_idxs.reshape(N, -1)

        # generate mask
        mask_shape = N, (size * down_factor) ** 2
        mask = torch.zeros(mask_shape, dtype=torch.int64, device=x.device)
        mask.scatter_(1, global_sampled_pos, 1)
        mask = mask.reshape(N, -1)

        # replace non masked area with index
        nonzero_indices = torch.nonzero(mask)
        indices = torch.arange(1, L + 1, dtype=torch.int64, device=x.device).unsqueeze(0).expand(N, -1).reshape(-1)
        ids_restore = mask.index_put((nonzero_indices[:, 0], nonzero_indices[:, 1]), indices)

        return ids_restore

    def forward_encoder(self, x, p):
        # embed patches
        # B C H W -> B HW D
        x = self.patch_embed(x)
        if self.use_pan:
            p = self.pan_embed(p)
        else:
            p = x

        ids_restore = self.get_global_pos(x)

        # apply Transformer blocks
        for i in range(len(self.blocks)):
            if i % 2 == 0:
                x, p = self.blocks[i](x, p)
            else:
                x, _ = self.blocks[i](x, x)

        return x, ids_restore

    def forward_bottleneck(self, x, ids_restore):
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], 1, 1)
        x = x.flatten(2).transpose(1, 2)
        x_ = torch.cat([mask_tokens, x], dim=1)  # no cls token
        # B H'W' D'
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        N, HW, C = x.shape
        H = W = int(HW**.5)
        x = x.permute(0, 2, 1).reshape(N, C, H, W)

        return x

    def forward_decoder(self, x, pan, up_ms):
        # B HW D'

        x = self.decoder_embed(x)
        p = self.pan_decoder_embed(pan)
        up_ms = self.upms_decoder_embed(up_ms)

        if self.use_ms:
            x = x + p + up_ms
        else:
            x = x + p

        for i in range(len(self.blocks)):
            if i % 2 == 0:
                x, p = self.dec_blocks[i](x, p)
            else:
                x, _ = self.dec_blocks[i](x, x)

        x = self.decoder_pred(x + p)

        return x

    def ensemble(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B // self.scatter_num, -1, H, W)
        res = self.ensembler(x)
        return res

    def forward(self, masked_ms, masked_pan, pan, up_ms):

        # duplicate channels
        masked_pan = masked_pan.repeat(1, up_ms.shape[1], 1, 1)
        pan = pan.repeat(1, up_ms.shape[1], 1, 1)

        # repeat at batch dimension
        masked_ms = torch.repeat_interleave(masked_ms, repeats=self.scatter_num, dim=0)
        masked_pan = torch.repeat_interleave(masked_pan, repeats=self.scatter_num, dim=0)
        pan = torch.repeat_interleave(pan, repeats=self.scatter_num, dim=0)
        up_ms_ = torch.repeat_interleave(up_ms, repeats=self.scatter_num, dim=0)

        # forward
        x, ids_restore = self.forward_encoder(masked_ms, masked_pan)
        x = self.forward_bottleneck(x, ids_restore)
        pred = self.forward_decoder(x, pan, up_ms_)

        # ensemle
        pred = self.ensemble(pred)

        return pred + up_ms
