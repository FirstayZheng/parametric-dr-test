# Parametric UMAP by pytorch

### Execute the `init_env.sh` first to initialize your environment

### execute command
``` shell
python main.py --config=./configs/{yourconfigfile}.yaml --dataset={yourdataset}
```
#### the image of successful execute

### To-do list
- [x] Do ablation study between GCDR and PUMAP.
- [x] Rewrite the `train` in `projection_3D.py`
- [x] Implement the cutting of KNN Graph
- [x] Some distortion occurred with different batch_size.
- [x] Add ISOMAP.
- [x] Rewrite loss function with `torch.nn.module`
- [x] Rewrite the `PUMAP_module` in `UMAPModel.py`
- [x] Unify the dtype of tensor. 

### The mindmap for rewriting the `train` in `projection_3D.py`
1. Implement a trainer with checkpoint-saving function so that we can store the intermeidate results.
2. Leverage the trainer to improve our arch.
