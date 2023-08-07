from dataset.originalDataset import ORI_DATASET
from base.base_model import FCEncoder
from audtorch.metrics.functional import pearsonr
import utils.visualization as visualization
import torch

def global_feature_loss(x_embeddings, to_x):
        # embedding_to, embedding_from = torch.chunk(x_embeddings, 2, dim=1)
        embedding_to = x_embeddings
        # print(embedding_to.shape)
        
        to_x = torch.flatten(to_x, start_dim=1)
        
        num = to_x.shape[0]
        high_matrix = to_x.repeat(num, 1, 1)
        low_matrix = embedding_to.repeat(num, 1, 1)

        high_dis = torch.norm(high_matrix - high_matrix.transpose(0, 1), dim=-1)
        low_dis = torch.norm(low_matrix - low_matrix.transpose(0, 1), dim=-1)

        corr = torch.mean(
            pearsonr(
                high_dis,
                low_dis,
            )[:, 0])
        return -corr


class myDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = torch.Tensor(data)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # print("index:", index)
        return self.data[index]



def linear_train(config):
    original_dataset = ORI_DATASET[config.dataset_category](config.data_path)
    X, color, n_items, input_dims = original_dataset.get_attr()

    model = FCEncoder(in_features=3)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    dataset = myDataset(X)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=256, shuffle=True)
    
    model.train()
    
    for i in range(40):
        for batch_idx, data in enumerate(dataloader):
            optimizer.zero_grad()  
            output = model(data)
            loss = global_feature_loss(output, data)
            loss.backward()
            optimizer.step()
            print(loss)
    
    final_projection = model(X).detach().numpy()
    
    visualization.plot_2D(final_projection,
                          color,
                          cmap='tab10',
                          savePath=config.output_dir,
                          title=config.plot_title)
    

if __name__ == "__main__":
    linear_train()
    