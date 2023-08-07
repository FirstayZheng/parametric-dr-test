# coding=utf-8
import torch
import torch.optim
import torch.utils.data
import numpy as np
import torch.nn as nn
from utils.graph import generate_graph
import utils.visualization as visualization
from dataset.edgeDataset import edgeDataset
from dataset.originalDataset import ORI_DATASET
import math
from model.loss import LOSS_MODE
# from model.loss_avg import LOSS_MODE
# from model.loss_grad import LOSS_MODE
from model.UMAPModel import UMAPModel
from trainer.trainer import Trainer
from utils.graph import find_ab_params, generate_graph
EPS = 1e-5


def train(config):
    # path = f'./data/{config.prefix}/{config.dataset}.npy'
    path = config.data_path
    original_dataset = ORI_DATASET[config.dataset_category](path)
    X, color, n_items, input_dims = original_dataset.get_attr()
    
    print(X.shape)
    # exit()
    
    # len_epoc= 
    config.total_steps = config.epochs * 1000

    umap_graph = generate_graph(X, config)
    # fuzzy simplical set
    
    # print(umap_graph)
    edge_dataset = edgeDataset(X, graph_=umap_graph)
    
    edge_dataloader = torch.utils.data.DataLoader(
       dataset=edge_dataset,
       batch_size=config.batch_size,
       shuffle=True,
       # num_workers=10,
       drop_last=True
    )
    
    model = UMAPModel(input_dims=input_dims).to(config.device)

    _a, _b = find_ab_params(1.0, min_dist=0.1)
    loss = LOSS_MODE[config.loss_mode](
        batch_size=config.batch_size,
        negative_sample_rate=config.negative_sample_rate,
        _a=_a,
        _b=_b,
        device=config.device,
        global_loss=config.global_loss,
        total_steps=config.total_steps,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    trainer = Trainer(model=model,
                      original_data=X.to(config.device),
                      criterion=loss,
                      optimizer=optimizer,
                      config=config,
                      device=config.device,
                      data_loader=edge_dataloader,
                      len_epoch=1000,
                      draw_gap=1,
                      X=X)

    trainer.train()

    # plot results
    result_list = trainer.result_list

    visualization.plot_small_multiples(points_list=result_list,
                                       ncols=10,
                                       nrows=4,
                                       color=color,
                                       title=f'{config.plot_title}-all',
                                       savePath=config.output_dir)

    if config.save_procedure_data:
        for p, (batch_idx, data) in enumerate(result_list):
            _path = config.output_dir
            _path = '/'.join(_path.split('/'))
            np.save(f'{_path}/procedure_{p}.npy', data)


    final_projection = model.project(X.to(config.device)).to('cpu')

    # visualization.plot_original_3D(
    #     X,
    #     color,
    #     cmap='tab10',
    #     savePath=config.output_dir,
    #     title=config.plot_title,
    # )

    visualization.plot_2D(final_projection,
                          color,
                          cmap='tab10',
                          savePath=config.output_dir,
                          title=config.plot_title)
    # draw loss curve
    loss_curve = trainer.loss_value
    visualization.plot_loss_curve(
        loss_curve,
        savePath=config.output_dir,
    )
    
    if config.save_result:
        np.save(f'{config.output_dir}/final_projection.npy', final_projection)
