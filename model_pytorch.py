import torch
from torch import nn
import torch.nn.functional as F
from GradientReversalLayer_torch import ReverseLayerF


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.ReLU(True),

            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3, stride=1),
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1),
            nn.BatchNorm2d(48),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU(True)
        )

        self.class_classifier = nn.Sequential(
            nn.Linear(48 * 4 * 4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Dropout2d(),

            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Dropout2d(),

            nn.Linear(100, 10),
            nn.LogSoftmax()
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(48 * 4 * 4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),

            nn.Linear(100, 2),
            nn.LogSoftmax()
        )

    def forward(self, input_data, alpha):
        feature = self.feature_extractor(input_data)
        feature = torch.flatten(feature, 1)
        reversed_features = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reversed_features)
        return class_output, domain_output

