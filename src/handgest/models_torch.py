"""Two temporal encoders share the same keypoint input contract."""

from torch import nn

from . import LABELS
from .features import WIDTH


class GRUClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(WIDTH)
        self.encoder = nn.GRU(
            WIDTH, 128, num_layers=2, dropout=0.3, bidirectional=True, batch_first=True
        )
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, len(LABELS)))

    def forward(self, x):
        states, _ = self.encoder(self.norm(x))
        return self.head(states.mean(dim=1))


class TemporalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        channels = WIDTH
        for dilation in (1, 2, 4):
            layers += [
                nn.Conv1d(channels, 128, 3, dilation=dilation, padding=dilation),
                nn.BatchNorm1d(128),
                nn.GELU(),
                nn.Dropout(0.2),
            ]
            channels = 128
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(channels, len(LABELS))

    def forward(self, x):
        return self.head(self.encoder(x.transpose(1, 2)).mean(dim=-1))


MODELS = {"gru": GRUClassifier, "tcn": TemporalCNN}
