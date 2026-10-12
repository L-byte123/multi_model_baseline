from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F


class TinySignalEncoder(nn.Module):
    """Offline smoke-test encoder, not the research backbone."""
    output_dim = 32

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                                    nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, x):
        return self.layers(x)


class PaSSTEncoder(nn.Module):
    output_dim = 768

    def __init__(self, cfg, initialize_pretrained):
        super().__init__()
        try:
            from hear21passt.models.passt import get_model
        except ImportError as error:
            raise ImportError('Install PaSST extras: pip install -e ".[passt,text]"') from error
        # Same architecture/stride as d25_t6/passt.py. Replace its audio Mel
        # frontend with the vibration STFT frontend, keeping frequency order.
        self.backbone = get_model(arch="passt_s_p16_s16_128_ap468",
            pretrained=initialize_pretrained and cfg.pretrained_signal,
            input_fdim=128, input_tdim=cfg.width, fstride=16, tstride=16,
            s_patchout_t=0, s_patchout_f=0)

    def forward(self, x):
        return self.backbone(x)[-1]


class ResNetEncoder(nn.Module):
    output_dim = 512

    def __init__(self, cfg, initialize_pretrained):
        super().__init__()
        try:
            from torchvision.models import ResNet18_Weights, resnet18
        except ImportError as error:
            raise ImportError('Install image extras: pip install -e ".[image,text]"') from error
        weights = ResNet18_Weights.DEFAULT if initialize_pretrained and cfg.pretrained_signal else None
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        return self.backbone(x)


class TextEncoder(nn.Module):
    def __init__(self, cfg, saved_config=None):
        super().__init__()
        from transformers import AutoConfig, AutoModel
        if saved_config is None:
            self.backbone = AutoModel.from_pretrained(cfg.text_model,
                local_files_only=cfg.local_files_only, trust_remote_code=False)
        else:
            params = dict(saved_config)
            model_type = params.pop("model_type")
            self.backbone = AutoModel.from_config(AutoConfig.for_model(model_type, **params))
        self.output_dim = self.backbone.config.hidden_size

    def forward(self, tokens):
        # DCASE's first-token pooling; retain decimal points and units in prompts.
        return self.backbone(**tokens).last_hidden_state[:, 0]


class FaultClassifier(nn.Module):
    def __init__(self, cfg, num_classes, numeric_dim=0, saved_text_config=None,
                 initialize_pretrained=True):
        super().__init__()
        self.cfg = cfg.validate()
        self.signal_encoder = None
        self.text_encoder = None
        branch_count = 0
        if cfg.fusion != "text_only":
            encoders = {"passt": PaSSTEncoder, "resnet18": ResNetEncoder}
            self.signal_encoder = (TinySignalEncoder() if cfg.signal_encoder == "tiny" else
                                   encoders[cfg.signal_encoder](cfg, initialize_pretrained))
            self.signal_projection = nn.Linear(self.signal_encoder.output_dim, cfg.projection_dim)
            branch_count += 1
        if cfg.fusion in {"text", "text_only"}:
            self.text_encoder = TextEncoder(cfg, saved_text_config)
            self.text_projection = nn.Linear(self.text_encoder.output_dim, cfg.projection_dim)
            branch_count += 1
        if cfg.fusion == "numeric":
            if numeric_dim < 1:
                raise ValueError("Numeric fusion requires numeric_dim")
            self.numeric_projection = nn.Linear(numeric_dim, cfg.projection_dim)
            branch_count += 1
        self.classifier = nn.Sequential(nn.Linear(branch_count*cfg.projection_dim, cfg.hidden_dim),
            nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.hidden_dim, num_classes))
        for encoder, frozen in ((self.signal_encoder, cfg.freeze_signal), (self.text_encoder, cfg.freeze_text)):
            if encoder is not None and frozen:
                encoder.requires_grad_(False)
                encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        # Frozen backbones must not update BatchNorm or use stochastic dropout.
        for encoder, frozen in ((self.signal_encoder, self.cfg.freeze_signal),
                                (self.text_encoder, self.cfg.freeze_text)):
            if encoder is not None and frozen:
                encoder.eval()
        return self

    def forward(self, visual, tokens=None, numeric=None):
        features = []
        if self.signal_encoder is not None:
            with torch.no_grad() if self.cfg.freeze_signal else nullcontext():
                signal = self.signal_encoder(visual)
            features.append(F.normalize(self.signal_projection(signal), dim=-1))
        if self.text_encoder is not None:
            if tokens is None:
                raise ValueError("Text fusion requires tokenized equipment descriptions")
            with torch.no_grad() if self.cfg.freeze_text else nullcontext():
                text = self.text_encoder(tokens)
            features.append(F.normalize(self.text_projection(text), dim=-1))
        if self.cfg.fusion == "numeric":
            features.append(F.normalize(self.numeric_projection(numeric), dim=-1))
        return self.classifier(torch.cat(features, dim=-1))

    def text_config(self):
        return self.text_encoder.backbone.config.to_dict() if self.text_encoder is not None else None
