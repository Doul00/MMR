
import yaml
import torch
import pytorch_lightning as pl

from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

from model.model import ToolSegmenterModel

class ToolSegmenterModule(pl.LightningModule):
    def __init__(self, config_path: str):
        self.config = yaml.safe_load(open(config_path, 'r'))
        self.model = ToolSegmenterModel(**self.config['model_opts'])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = MNIST(root='data', train=True, download=True, transform=transform)
        return DataLoader(dataset, batch_size=32, shuffle=True)