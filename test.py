from umap_pytorch import PUMAP
import torch
from dataset.originalDataset import generateSwissrollDataset
from utils.visualization import plot_2D
pumap = PUMAP(
        encoder=None,           # nn.Module, None for default
        decoder=None,           # nn.Module, True for default, None for encoder only
        n_neighbors=10,
        min_dist=0.1,
        metric="euclidean",
        n_components=2,
        beta=1.0,               # How much to weigh reconstruction loss for decoder
        reconstruction_loss=F.binary_cross_entropy_with_logits, # pass in custom reconstruction loss functions
        random_state=None,
        lr=1e-3,
        epochs=10,
        batch_size=64,
        num_workers=1,
        num_gpus=1,
        match_nonparametric_umap=False # Train network to match embeddings from non parametric umap
)


original_dataset = generateSwissrollDataset("./data/swissroll/swissroll.npy")
data, color, n_items, input_dims = original_dataset.get_attr()
# data = torch.randn((50000, 512))
pumap.fit(data)
embedding = pumap.transform(data) # (50000, 2)

plot_2D(embedding, color=color, title="test")
# def plot_2D(points, color=None,cmap=None, savePath='./outputs', title=None):
#     x, y = points.T
#     fig, ax = plt.subplots()
#     ax.scatter(x, y, s=1, c=color, cmap=cmap)
#     ax.set_title(title)
#     if not os.path.exists(f'{savePath}'):
#         os.makedirs(f'{savePath}')
#     fig.savefig(f'{savePath}/{title}.png')
# if decoder enabled
# recon = pumap.inverse_transform(embedding)  # (50000, 512)