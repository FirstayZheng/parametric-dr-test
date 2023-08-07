import abc
import torch
import sklearn.datasets
import numpy as np


class baseDataset(abc.ABC):
    def __init__(self, path=None):
        self.original_data = None
        self.path = path
        self.n_items = None
        self.input_dims = None
        self.color = None
        self.X = None
        self.load_data()
    
    def misc(self):
        self.X = torch.tensor(
            self.original_data,
            dtype=torch.float32,
        ).to('cpu')# force dtype to be torch.float32, which is also the defautl prama of pytorch model

        self.n_items = self.X.shape[0]
        self.input_dims = self.X.shape[-1]

    def get_attr(self):
        return self.X, self.color, self.n_items, self.input_dims,

    @abc.abstractmethod
    def load_data(self,):
        pass


class localDataset(baseDataset):

    def __init__(self, path=None):
        super().__init__(path)
        self.color = np.linspace(0, 10 - 1e3, self.n_items, dtype=int)
    
    def load_data(self,):
        self.original_data = np.load(self.path)
        self.misc()


class generateSwissrollDataset(baseDataset):

    def __init__(self, path=None):
        super().__init__()

    def load_data(self,):
        self.original_data, self.color = \
            sklearn.datasets.make_swiss_roll(
                n_samples=4500,
                random_state=None
            )
        self.misc()


class faceDataset(baseDataset):
    def __init__(self, path:str=None):
        path = path.split('.')
        path[-1] = 'csv'
        path = '.'.join(path)
        super().__init__(path)

    def load_data(self):
        self.original_data = np.loadtxt(self.path, delimiter=',', dtype=float)
        self.misc()



class generateSCurveDataset(baseDataset):

    def __init__(self, path=None):
        super().__init__()

    def load_data(self,):
        self.original_data, self.color = \
            sklearn.datasets.make_s_curve(
                n_samples=4500,
                random_state=None
            )
        self.misc()

# register datasetclass
ORI_DATASET = {
    'local': localDataset,
    'swissroll': generateSwissrollDataset,
    'face': faceDataset,
    's_curve':generateSCurveDataset,
}