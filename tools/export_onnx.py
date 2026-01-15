#!/usr/bin/env python3
import argparse, torch, onnx
from mmengine import Config
from mmdet.apis import init_detector

# ثبت ماژول‌های اختصاصی CLRerNet
import libs.models
import libs.core.bbox
import libs.core.anchor
import libs.models.dense_heads.clrernet_head
import libs.models.detectors.clrernet
import libs.models.backbones.dla
import libs.models.layers.attentions


def parse_args():
    parser = argparse.ArgumentParser(description="Export CLRerNet to ONNX")
    parser.add_argument("--config", required=True, help="Path to config file (.py)")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth file")
    parser.add_argument("--output", default="clrernet.onnx", help="Output ONNX file")
    parser.add_argument("--device", default="cpu", help="Device for export (cpu/cuda)")
    parser.add_argument("--opset", type=int, default=16, help="ONNX opset version")
    return parser.parse_args()


class Wrapper(torch.nn.Module):
    def __init__(self, det):
        super().__init__()
        self.det = det

    def forward(self, x):
        feats = self.det.extract_feat(x)
        out_list = self.det.bbox_head.forward(feats)
        last = out_list[-1]
        return last["cls_logits"], last["anchor_params"], last["lengths"], last["xs"]


def main():
    args = parse_args()

    print(f"🔧 Loading model from {args.checkpoint}")
    model = init_detector(args.config, args.checkpoint, device=args.device)
    model.eval()
    wrapper = Wrapper(model)

    dummy = torch.randn(1, 3, 320, 800, device=args.device)
    torch.onnx.export(
        wrapper,
        dummy,
        args.output,
        opset_version=args.opset,
        input_names=["input"],
        output_names=["cls_logits", "anchor_params", "lengths", "xs"],
        dynamic_axes={"input": {0: "batch"}},
    )
    onnx.load(args.output)
    print(f"✅ Export complete: {args.output}")


if __name__ == "__main__":
    main()
