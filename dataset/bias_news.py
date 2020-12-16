import torch

from torch.utils.data import Dataset
from torchvision.transforms import transforms


class BiasNews(Dataset):

    def __init__(self, sequences, labels):
        # TODO
        self.sequences = sequences
        self.labels = labels
        self.transform = transforms.ToTensor()

    def __getitem__(self, index):
        sequence = torch.tensor(self.sequences[index], dtype=torch.long)
        # sequences = torch.squeeze(self.transform(np.array(self.sequences[index], dtype=np.float32)))
        label = self.labels[index]
        label = 1 if label else 0
        return sequence, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.labels)
