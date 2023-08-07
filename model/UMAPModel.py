import torch
from base.base_model import BaseModel, FCEncoder


class UMAPModel(BaseModel):
    def __init__(self, input_dims) -> None:
        super().__init__()
        self.input_dims = input_dims
        self.encoder = FCEncoder(input_dims)  
        # self.encoder = nn.DataParallel(FCEncoder(input_dims), device_ids=[0, 1], output_device=0)
        
    def project(self, input):
        # project data to 2D
        output = None
        with torch.no_grad():
            output = self.encoder(input)
        return output
        
    def forward(self, to_x, from_x):
        tx = self.encoder(to_x)
        fx = self.encoder(from_x)
        rlt = torch.cat([tx, fx], dim=1)
        return rlt
