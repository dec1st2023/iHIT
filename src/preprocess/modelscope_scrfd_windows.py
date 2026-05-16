from __future__ import annotations

import os
from pathlib import Path

import numpy as np


MODEL_ID = "damo/cv_resnet_facedetection_scrfd10gkps"


def check_runtime() -> None:
    try:
        import torch  # noqa: F401
        from modelscope import snapshot_download as _snapshot_download  # noqa: F401
        from PIL import Image as _Image  # noqa: F401
    except Exception as exc:  # pragma: no cover - dependency message only
        raise RuntimeError("Windows ModelScope SCRFD requires torch, Pillow, and modelscope.") from exc


def build_detector(model_dir: str | Path | None = None, device: str | None = None):
    check_runtime()
    if model_dir is None:
        from modelscope import snapshot_download

        model_dir = snapshot_download(MODEL_ID)
    return ModelScopeWindowsSCRFD(Path(model_dir), device=device)


def _torch_nn_module():
    import torch.nn as nn

    return nn.Module


class ModelScopeWindowsSCRFD:
    """Windows-friendly SCRFD inference for the ModelScope SCRFD checkpoint.

    ModelScope's official SCRFD pipeline uses mmdet/mmcv-full. On Windows with
    modern PyTorch that extension stack is often unavailable, so this adapter
    loads the same ModelScope checkpoint into a small PyTorch inference graph.
    """

    def __init__(self, model_dir: Path, device: str | None = None, score_thr: float = 0.3, nms_iou: float = 0.45, max_per_img: int = 1000):
        import torch

        self.model_dir = Path(model_dir)
        self.score_thr = score_thr
        self.nms_iou = nms_iou
        self.max_per_img = max_per_img
        if device is None:
            device = os.environ.get("IHIT_SCRFD_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.model = _SCRFDDetector().to(self.device)
        checkpoint_path = self.model_dir / "pytorch_model.bin"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"ModelScope SCRFD checkpoint not found: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - for older torch versions
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    def __call__(self, image_path: str | Path) -> dict[str, list]:
        import torch

        tensor, meta = _preprocess_image(Path(image_path), self.device)
        with torch.inference_mode():
            cls_scores, bbox_preds, kps_preds = self.model(tensor)
            boxes, scores, keypoints = _decode_outputs(
                cls_scores,
                bbox_preds,
                kps_preds,
                meta,
                score_thr=self.score_thr,
                nms_iou=self.nms_iou,
                max_per_img=self.max_per_img,
            )
        return {
            "boxes": boxes.detach().cpu().numpy().tolist(),
            "scores": scores.detach().cpu().numpy().tolist(),
            "keypoints": keypoints.detach().cpu().numpy().tolist(),
        }


class _SCRFDDetector(_torch_nn_module()):
    def __init__(self):
        super().__init__()
        self.backbone = _ResNetV1e()
        self.neck = _PAFPN()
        self.bbox_head = _SCRFDHead()

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        return self.bbox_head(x)


class _ResNetV1e(_torch_nn_module()):
    def __init__(self):
        import torch.nn as nn

        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 28, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(28),
            nn.ReLU(inplace=True),
            nn.Conv2d(28, 28, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(28),
            nn.ReLU(inplace=True),
            nn.Conv2d(28, 56, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(56),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer1 = _make_res_layer(56, 56, 17, stride=1)
        self.layer2 = _make_res_layer(224, 56, 16, stride=2)
        self.layer3 = _make_res_layer(224, 144, 2, stride=2)
        self.layer4 = _make_res_layer(576, 184, 8, stride=2)

    def forward(self, x):
        x = self.stem(x)
        x = self.maxpool(x)
        outs = []
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            x = layer(x)
            outs.append(x)
        return tuple(outs)


class _Bottleneck(_torch_nn_module()):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample=None):
        import torch.nn as nn

        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return self.relu(out)


def _make_res_layer(inplanes: int, planes: int, blocks: int, stride: int):
    import torch.nn as nn

    downsample = None
    outplanes = planes * _Bottleneck.expansion
    if stride != 1 or inplanes != outplanes:
        downsample = nn.Sequential(
            nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False),
            nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(outplanes),
        )
    layers = [_Bottleneck(inplanes, planes, stride=stride, downsample=downsample)]
    layers.extend(_Bottleneck(outplanes, planes, stride=1) for _ in range(1, blocks))
    return nn.Sequential(*layers)


class _ConvOnly(_torch_nn_module()):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        import torch.nn as nn

        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=True)

    def forward(self, x):
        return self.conv(x)


class _ConvGNReLU(_torch_nn_module()):
    def __init__(self, in_channels: int, out_channels: int):
        import torch.nn as nn

        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.gn = nn.GroupNorm(32, out_channels)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activate(self.gn(self.conv(x)))


class _PAFPN(_torch_nn_module()):
    def __init__(self):
        import torch.nn as nn

        super().__init__()
        self.lateral_convs = nn.ModuleList(
            [
                _ConvOnly(224, 128, 1),
                _ConvOnly(576, 128, 1),
                _ConvOnly(736, 128, 1),
            ]
        )
        self.fpn_convs = nn.ModuleList([_ConvOnly(128, 128, 3, padding=1) for _ in range(3)])
        self.downsample_convs = nn.ModuleList([_ConvOnly(128, 128, 3, stride=2, padding=1) for _ in range(2)])
        self.pafpn_convs = nn.ModuleList([_ConvOnly(128, 128, 3, padding=1) for _ in range(2)])

    def forward(self, inputs):
        import torch.nn.functional as F

        laterals = [conv(inputs[index + 1]) for index, conv in enumerate(self.lateral_convs)]
        for index in range(len(laterals) - 1, 0, -1):
            laterals[index - 1] = laterals[index - 1] + F.interpolate(laterals[index], size=laterals[index - 1].shape[2:], mode="nearest")
        inter_outs = [conv(lateral) for conv, lateral in zip(self.fpn_convs, laterals)]
        for index, downsample in enumerate(self.downsample_convs):
            inter_outs[index + 1] = inter_outs[index + 1] + downsample(inter_outs[index])
        return (inter_outs[0], self.pafpn_convs[0](inter_outs[1]), self.pafpn_convs[1](inter_outs[2]))


class _Scale(_torch_nn_module()):
    def __init__(self):
        import torch
        import torch.nn as nn

        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, x):
        return x * self.scale


class _Integral(_torch_nn_module()):
    def __init__(self, reg_max: int = 8):
        import torch

        super().__init__()
        self.register_buffer("project", torch.linspace(0, reg_max, reg_max + 1))


class _SCRFDHead(_torch_nn_module()):
    def __init__(self):
        import torch.nn as nn

        super().__init__()
        self.cls_stride_convs = nn.ModuleDict({"0": nn.ModuleList([_ConvGNReLU(128, 256), _ConvGNReLU(256, 256)])})
        self.reg_stride_convs = nn.ModuleDict({"0": nn.ModuleList()})
        self.stride_cls = nn.ModuleDict({"0": nn.Conv2d(256, 2, 3, padding=1)})
        self.stride_reg = nn.ModuleDict({"0": nn.Conv2d(256, 8, 3, padding=1)})
        self.stride_kps = nn.ModuleDict({"0": nn.Conv2d(256, 20, 3, padding=1)})
        self.scales = nn.ModuleList([_Scale(), _Scale(), _Scale()])
        self.integral = _Integral()

    def forward(self, feats):
        cls_scores = []
        bbox_preds = []
        kps_preds = []
        convs = self.cls_stride_convs["0"]
        for feat, scale in zip(feats, self.scales):
            cls_feat = feat
            for conv in convs:
                cls_feat = conv(cls_feat)
            cls_scores.append(self.stride_cls["0"](cls_feat))
            bbox_preds.append(scale(self.stride_reg["0"](cls_feat)))
            kps_preds.append(self.stride_kps["0"](cls_feat))
        return cls_scores, bbox_preds, kps_preds


def _preprocess_image(path: Path, device):
    import torch
    from PIL import Image

    image = Image.open(path).convert("RGB")
    original_width, original_height = image.size
    scale = min(640.0 / original_width, 640.0 / original_height)
    resized_width = max(1, int(original_width * scale + 0.5))
    resized_height = max(1, int(original_height * scale + 0.5))
    image = image.resize((resized_width, resized_height), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32)
    padded = np.zeros((640, 640, 3), dtype=np.float32)
    padded[:resized_height, :resized_width, :] = array
    padded = (padded - 127.5) / 128.0
    tensor = torch.from_numpy(padded.transpose(2, 0, 1)).unsqueeze(0).to(device)
    meta = {
        "img_shape": (resized_height, resized_width, 3),
        "scale_factor": np.array([resized_width / original_width, resized_height / original_height] * 2, dtype=np.float32),
    }
    return tensor, meta


def _decode_outputs(cls_scores, bbox_preds, kps_preds, meta, score_thr: float, nms_iou: float, max_per_img: int):
    import torch

    all_boxes = []
    all_scores = []
    all_keypoints = []
    for cls_score, bbox_pred, kps_pred, stride, base_size in zip(cls_scores, bbox_preds, kps_preds, (8, 16, 32), (16, 64, 256)):
        scores = cls_score[0].permute(1, 2, 0).reshape(-1).sigmoid()
        boxes_delta = bbox_pred[0].permute(1, 2, 0).reshape(-1, 4) * stride
        keypoints_delta = kps_pred[0].permute(1, 2, 0).reshape(-1, 10) * stride
        anchors = _grid_anchors(cls_score.shape[-2:], stride, base_size, cls_score.device, cls_score.dtype)
        centers = _anchor_centers(anchors)
        boxes = _distance2bbox(centers, boxes_delta, meta["img_shape"])
        keypoints = _distance2kps(centers, keypoints_delta)
        keep = scores > score_thr
        if keep.any():
            all_boxes.append(boxes[keep])
            all_scores.append(scores[keep])
            all_keypoints.append(keypoints[keep])
    if not all_boxes:
        empty = torch.empty((0,), device=cls_scores[0].device)
        return torch.empty((0, 4), device=cls_scores[0].device), empty, torch.empty((0, 10), device=cls_scores[0].device)
    boxes = torch.cat(all_boxes, dim=0)
    scores = torch.cat(all_scores, dim=0)
    keypoints = torch.cat(all_keypoints, dim=0)
    scale = torch.as_tensor(meta["scale_factor"], device=boxes.device, dtype=boxes.dtype)
    boxes = boxes / scale
    keypoint_scale = torch.as_tensor([scale[0], scale[1]] * 5, device=boxes.device, dtype=boxes.dtype)
    keypoints = keypoints / keypoint_scale
    keep_indices = _nms(boxes, scores, nms_iou)
    if max_per_img > 0:
        keep_indices = keep_indices[:max_per_img]
    return boxes[keep_indices], scores[keep_indices], keypoints[keep_indices]


def _grid_anchors(featmap_size, stride: int, base_size: int, device, dtype):
    import torch

    feat_h, feat_w = featmap_size
    scales = torch.as_tensor([1.0, 2.0], device=device, dtype=dtype)
    widths = base_size * scales
    base = torch.stack([-0.5 * widths, -0.5 * widths, 0.5 * widths, 0.5 * widths], dim=-1)
    shift_x = torch.arange(0, feat_w, device=device, dtype=dtype) * stride
    shift_y = torch.arange(0, feat_h, device=device, dtype=dtype) * stride
    yy, xx = torch.meshgrid(shift_y, shift_x, indexing="ij")
    shifts = torch.stack([xx.reshape(-1), yy.reshape(-1), xx.reshape(-1), yy.reshape(-1)], dim=-1)
    return (base[None, :, :] + shifts[:, None, :]).reshape(-1, 4)


def _anchor_centers(anchors):
    import torch

    return torch.stack([(anchors[:, 0] + anchors[:, 2]) * 0.5, (anchors[:, 1] + anchors[:, 3]) * 0.5], dim=-1)


def _distance2bbox(points, distance, max_shape):
    import torch

    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    height, width = max_shape[:2]
    return torch.stack([x1.clamp(min=0, max=width), y1.clamp(min=0, max=height), x2.clamp(min=0, max=width), y2.clamp(min=0, max=height)], dim=-1)


def _distance2kps(points, distance):
    import torch

    preds = []
    for index in range(0, distance.shape[1], 2):
        preds.append(points[:, 0] + distance[:, index])
        preds.append(points[:, 1] + distance[:, index + 1])
    return torch.stack(preds, dim=-1)


def _nms(boxes, scores, iou_threshold: float):
    import torch

    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        index = order[0]
        keep.append(index)
        if order.numel() == 1:
            break
        current = boxes[index]
        rest = boxes[order[1:]]
        xx1 = torch.maximum(current[0], rest[:, 0])
        yy1 = torch.maximum(current[1], rest[:, 1])
        xx2 = torch.minimum(current[2], rest[:, 2])
        yy2 = torch.minimum(current[3], rest[:, 3])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        area_current = (current[2] - current[0]).clamp(min=0) * (current[3] - current[1]).clamp(min=0)
        area_rest = (rest[:, 2] - rest[:, 0]).clamp(min=0) * (rest[:, 3] - rest[:, 1]).clamp(min=0)
        iou = inter / (area_current + area_rest - inter + 1e-6)
        order = order[1:][iou <= iou_threshold]
    if not keep:
        return torch.empty((0,), device=boxes.device, dtype=torch.long)
    return torch.stack(keep)
