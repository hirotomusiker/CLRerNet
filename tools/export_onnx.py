import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))   

import torch, onnx
from mmengine import Config
from mmengine.registry import build_from_cfg
from mmdet.registry import MODELS
import mmdet.models
 
import libs.models
import libs.core.bbox
import libs.core.anchor
import libs.models.dense_heads.clrernet_head
import libs.models.detectors.clrernet
import libs.models.backbones.dla
import libs.models.layers.attentions  

 

CFG = "configs/clrernet/culane/clrernet_culane_dla34_ema.py"
CKPT = "clrernet_culane_dla34_ema.pth"
ONNX_OUT = "clrernet.onnx"


def build():
    cfg = Config.fromfile(CFG)
    if "data_preprocessor" in cfg.model:
        print("⚠️ Removing unsupported data_preprocessor for MMDet v2...")
        cfg.model.pop("data_preprocessor")

    if hasattr(cfg, "custom_imports"):
        cfg.custom_imports.imports = [
            m for m in cfg.custom_imports.imports if not m.startswith("libs.datasets")
        ]

    

    model = MODELS.build(cfg.model)
    sd = torch.load(CKPT, map_location="cpu")
    state = sd.get("state_dict", sd)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


class Wrapper(torch.nn.Module):
    def __init__(self, det):
        super().__init__()
        self.det = det

    def forward(self, x):
        feats = self.det.extract_feat(x)
        out_list = self.det.bbox_head.forward(feats)
        last = out_list[-1]
        return last["cls_logits"], last["anchor_params"], last["lengths"], last["xs"]


if __name__ == "__main__":
    model = build()
    wrapper = Wrapper(model)
    dummy = torch.randn(1, 3, 320, 800)
    torch.onnx.export(
        wrapper,
        dummy,
        ONNX_OUT,
        opset_version=16,
        input_names=["input"],
        output_names=["cls_logits", "anchor_params", "lengths", "xs"],
        dynamic_axes={"input": {0: "b"}},
    )
    onnx.load(ONNX_OUT)
    print("✅ Exported:", ONNX_OUT)
