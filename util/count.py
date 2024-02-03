
import torch
from models_mae import *
from thop import profile
from thop import clever_format


def count(model, e, name):
    model = model(img_size=16, ms_chans=8, ensemble=e, pos_type="2d_sincos", attn_type="naive")
    ms = torch.randn(1, 8, 16, 16)
    up_ms = torch.randn(1, 8, 64, 64)
    pan = torch.randn(1, 1, 64, 64)

    macs, params = profile(model, inputs=(ms, pan, up_ms))
    macs, params = clever_format([macs, params], "%.3f")
    print(name + "-"*100 + str(e))
    print(macs)
    print(params)



count(mae_vit_tiny, 1, "mae_vit_tiny")
count(mae_vit_tiny, 2, "mae_vit_tiny")
count(mae_vit_tiny, 4, "mae_vit_tiny")
count(mae_vit_tiny, 6, "mae_vit_tiny")
count(mae_vit_tiny, 8, "mae_vit_tiny")
