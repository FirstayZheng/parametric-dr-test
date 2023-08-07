import torch.nn as nn
import torch
import numpy as np
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])  # type: ignore
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
    
    def predict(self, x):
        with torch.no_grad():
            y = self(x)
        return y


class FCEncoder(nn.Module):
    def __init__(self, in_features, hidden_dims=None, n_components=2):
        """

        Args:
            in_features (_type_): shape of input
            hidden_dims (_type_, optional): _description_. Defaults to None.
            n_components (int, optional): number of latent dimensions. Defaults to 2.
        """
        nn.Module.__init__(self)

        if hidden_dims is None:
            hidden_dims = [256, 512, 512, 1024]
            # 小规模数据集上使用较浅层网络
            # hidden_dims = [128, 256, 512]
            # hidden_dims = [64, 128, 256]
            # hidden_dims = [32, 64, 128]
            # hidden_dims = [16, 32, 64]
            # hidden_dims = [128, 256]
        self.hidden_dims = hidden_dims
        modules = []

        in_dim = in_features
        for dim in hidden_dims:
            modules.append(
                nn.Sequential(nn.Linear(in_dim, dim), nn.ReLU()))
            in_dim = dim
        self.encoder = nn.Sequential(*modules, nn.Linear(in_dim, n_components))

    def forward(self, x):
        return self.encoder(x)

    def predict(self, x):
        with torch.no_grad():
            y = self(x)
        self.train()
        return y

