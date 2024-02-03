from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block
from util.sparse import Block as DecBlock
from util.patch_embed import PanEmed, PatchEmbed
from util.pos_embed import create_trainable_embeddings, get_2d_sincos_pos_embed

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, kernel_size, kernel_num):
        super(ResidualBlock, self).__init__()
        self.Conv1 = nn.Conv2d(in_channels=in_channels, padding=1, kernel_size=kernel_size, out_channels=kernel_num)
        self.Conv2 = nn.Conv2d(in_channels=in_channels, padding=1, kernel_size=kernel_size, out_channels=kernel_num)

    def forward(self, x):
        y = F.relu(self.Conv1(x), False)
        y = self.Conv2(y)
        return x + y


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    
    def __init__(self, img_size=16, ms_chans=4, ensemble=4, embed_dim=1024, down_factor=4, 
                 depth=24, num_heads=16, decoder_embed_dim=512, decoder_depth=8, 
                 decoder_num_heads=16, mlp_ratio=4., norm_layer=nn.LayerNorm,
                 pos_type="2d_sincos", attn_type="sparse"):
        super().__init__()
        self.img_size = img_size
        self.down_factor = down_factor
        self.pos_type = pos_type
        self.embed_dim = embed_dim
        self.attn_type = attn_type
        self.decoder_embed_dim = decoder_embed_dim
        
        # MAE encoder specifics
        # BCHW -> B HW D
        self.patch_embed = PatchEmbed(img_size, ms_chans, embed_dim)

        # 1 1 D
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 1 HW D
        if pos_type == "2d_sincos":
            self.pos_embed = nn.Parameter(torch.zeros(1, (img_size * down_factor) ** 2, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        elif pos_type == "trainable":
            self.pos_embed = nn.Parameter(torch.zeros(1, (img_size * down_factor) ** 2, embed_dim), requires_grad=True)  # trainable 1d embedding
        
        self.pan_embed = PanEmed(img_size, embed_dim)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)

        # MAE decoder specifics
        # B HW D -> B HW D'
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_pan_embed = PanEmed(img_size, decoder_embed_dim, 1)

        # 1 1 D'
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # 1 HW D'
        if pos_type == "2d_sincos":
            requires_grad = False
        elif pos_type == "trainable":
            requires_grad = True
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, (img_size * down_factor) ** 2, decoder_embed_dim), requires_grad=requires_grad)

        if attn_type == "sparse":
            blk_fn = DecBlock
        elif attn_type == "naive":
            blk_fn = Block
        self.decoder_blocks = nn.ModuleList([
            blk_fn(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, ms_chans, bias=True) # decoder to patch

        # MAE ensemble specifics
        self.scatter_num = ensemble
        kernel_num = ms_chans * ensemble
        kernel_size = 3
        self.ensembler = nn.Sequential(ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                       ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                       ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                       nn.Conv2d(in_channels=kernel_num, out_channels=ms_chans, padding=1, kernel_size=kernel_size)
                                       )
        self.initialize_weights()


    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        if self.pos_type == "2d_sincos":
            print("pos embed type: 2d_sincos.")
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.img_size * self.down_factor, cls_token=False)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
            decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.img_size * self.down_factor, cls_token=False)
            self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        elif self.pos_type == "trainable":
            print("pos embed type: trainable.")
            n_pos = torch.arange((self.img_size * self.down_factor) ** 2, dtype=torch.float32).cuda()
            pos_embed = create_trainable_embeddings(n_pos, self.embed_dim)
            self.pos_embed.data.copy_(pos_embed.float().unsqueeze(0))
            decoder_pos_embed = create_trainable_embeddings(n_pos, self.decoder_embed_dim)
            self.decoder_pos_embed.data.copy_(decoder_pos_embed.float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        w = self.pan_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        w = self.decoder_pan_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def patchify(self, imgs, p=None):
        """
        imgs: (N, C, H, W)
        x: (N, H*W, C)
        """
        N, C, H, W = imgs.shape
        x = imgs.reshape(N, C, -1).permute(0, 2, 1) # N HW MS
        
        if p is not None:
            N, C, H, W = p.shape
            p = p.reshape(N, C, -1).permute(0, 2, 1) # N HW 1
            return x, p
        
        return x


    def unpatchify(self, x):
        """
        x: (N, H'W', C)
        imgs: (N, C, H', W')
        """
        N, HW, C = x.shape
        H = W = int(HW**.5)
        imgs = x.permute(0, 2, 1).reshape(N, C, H, W)

        return imgs


    def get_global_pos(self, x):
        N, L, *_ = x.shape
        size = self.img_size
        down_factor = self.down_factor
        
        row = torch.arange(0, size * down_factor, down_factor, device=x.device).unsqueeze(-1)
        col = size * down_factor * down_factor * torch.arange(size, device=x.device)
        corner = (row + col).T # top left
        offset = torch.randint(down_factor, (N, size, size, 2), device=x.device) # offset to right and bottom
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

        return global_sampled_pos, mask, ids_restore # B HW , B H'W'
    
    def forward_encoder(self, x, p):
        
        # embed patches
        # B C H W -> B HW D
        x = self.patch_embed(x)
        x = x + self.pan_embed(p)
        
        # gather sampled pos embed
        # B HW
        global_sampled_pos, mask, ids_restore = self.get_global_pos(x) 
        # B HW D
        gathered_embedding = torch.gather(self.pos_embed.repeat(x.shape[0], 1, 1), dim=1, index=global_sampled_pos.unsqueeze(-1).repeat(1, 1, self.pos_embed.shape[-1]))

        # add pos embed w/o cls token
        # B HW D
        x = x + gathered_embedding

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore, global_sampled_pos
    
    
    def forward_decoder(self, x, ids_restore, pan, global_sampled_pos):
        # embed tokens
        # B HW D'
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], 1, 1)
        x_ = torch.cat([mask_tokens, x], dim=1)  # no cls token
        # B H'W' D'
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        # add pos embed
        x = x + self.decoder_pos_embed
        x = x + self.decoder_pan_embed(pan)
        
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            if self.attn_type == "sparse":
                x = blk(x, global_sampled_pos)
            elif self.attn_type == "naive":
                x = blk(x)
        
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)
        
        return x # N, H'W', C


    def ensemble(self, pred):
        B, C, H, W = pred.shape
        pred = pred.reshape(B // self.scatter_num, -1, H, W)
        pred = self.ensembler(pred)
        return pred
    
    
    def forward(self, imgs, p, up_ms):
        imgs = torch.repeat_interleave(imgs, repeats=self.scatter_num, dim=0)
        p = torch.repeat_interleave(p, repeats=self.scatter_num, dim=0)
        latent, mask, ids_restore, global_sampled_pos = self.forward_encoder(imgs, p)
        pred = self.forward_decoder(latent, ids_restore, p, global_sampled_pos)
        pred = self.unpatchify(pred)
        pred = self.ensemble(pred) + up_ms
        return pred, mask, global_sampled_pos
    

    

def mae_vit_tiny(**kwargs):
    model = MaskedAutoencoderViT(embed_dim=64, depth=1, num_heads=1,
        decoder_embed_dim=128, decoder_depth=1, decoder_num_heads=4,
        mlp_ratio=1, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_small(**kwargs):
    model = MaskedAutoencoderViT(embed_dim=128, depth=1, num_heads=2,
        decoder_embed_dim=128, decoder_depth=2, decoder_num_heads=4,
        mlp_ratio=1, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base(**kwargs):
    model = MaskedAutoencoderViT(embed_dim=256, depth=1, num_heads=4,
        decoder_embed_dim=128, decoder_depth=4, decoder_num_heads=4,
        mlp_ratio=1, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_cust(**kwargs):
    model = MaskedAutoencoderViT(embed_dim=64, depth=1, num_heads=1, decoder_num_heads=4,
        mlp_ratio=1, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
