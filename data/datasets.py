"""
@brief: Dataset utils for SARRARP50 dataset

"""
import pytorch_lightning as pl
from os.path import basename
from torch.utils.data import Dataset

from data.transforms import get_transforms


class SARRARP50Dataset(Dataset):
    def __init__(self, root_data_dir: str, listfile: str, config: dict):
        """
        Dataloader for SARRARP50 dataset.
        Here we load the samples from disk during __getitem__ as the whole training set will not fit in RAM.
        Using multiple worker processes will help.

        Args:
            root_data_dir (str): Root directory of the dataset containing the video data.
            listfile (str): Each element in the list has the format "video_yyy/rgb/xxxxxxx.png"
            config (dict): Dataset config
        """
        self.config = config
        self.transforms = get_transforms(self.config['transforms'])
        self.root_data_dir = root_data_dir
        self.index = self._make_index(listfile)

    def _make_index(self, listfile: str):
        all_samples = [x.strip() for x in open(listfile, 'r').readlines()]
        index = []
        for sample in all_samples:
            video_folder = sample.split('/')[0]
            img = f"{self.root_data_dir}/{sample}"
            label = f"{self.root_data_dir}/{video_folder}/segmentation/{basename(sample)}"
            index.append((img, label))
        return index

    def __getitem__(self, index):
        img, label = self.index[index]

    def __len__(self):
        return super().__len__()


class DataModule(pl.LightningDataModule):
    def __init__(self, config_path: str):
        super().__init__()
        self.config = yaml.safe_load(open(config_path, 'r'))
        self.batch_size = self.config['training_opts']['batch_size']

    def train_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = MNIST(root='data', train=True, download=True, transform=transform)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = MNIST(root='data', train=False, download=True, transform=transform)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)