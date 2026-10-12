from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass
class Config:
    signal_encoder: str = "passt"
    input_kind: str = "spectrogram"
    text_model: str = "roberta-large"
    freeze_text: bool = True
    freeze_signal: bool = False
    pretrained_signal: bool = True
    fusion: str = "text"
    projection_dim: int = 256
    hidden_dim: int = 128
    dropout: float = 0.2
    max_tokens: int = 64
    height: int = 128
    width: int = 256
    epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 0.001
    backbone_lr: float = 0.00001
    weight_decay: float = 0.0001
    seed: int = 42
    workers: int = 0
    local_files_only: bool = False

    def validate(self):
        if self.signal_encoder not in {"passt", "resnet18", "tiny"}:
            raise ValueError("signal_encoder must be passt, resnet18, or tiny")
        if self.input_kind not in {"spectrogram", "png"}:
            raise ValueError("input_kind must be spectrogram or png")
        if self.fusion not in {"text", "image", "numeric", "text_only"}:
            raise ValueError("fusion must be text, image, numeric, or text_only")
        if self.signal_encoder == "passt" and self.input_kind != "spectrogram":
            raise ValueError("PaSST accepts numeric spectrograms; use resnet18 for PNG")
        if self.signal_encoder == "passt" and (self.height != 128 or not 32 <= self.width <= 1000):
            raise ValueError("PaSST requires height=128 and width between 32 and 1000")
        for field in ("projection_dim", "hidden_dim", "max_tokens", "height", "width", "epochs", "batch_size"):
            if getattr(self, field) < 1:
                raise ValueError(f"{field} must be positive")
        if self.learning_rate <= 0 or self.backbone_lr <= 0 or self.workers < 0:
            raise ValueError("Learning rates must be positive; workers must be nonnegative")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        return self

    @classmethod
    def read(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8"))).validate()

    def to_dict(self):
        return asdict(self)
