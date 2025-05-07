import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image as PILImage
import torchvision.transforms as T

class HFImageDataset(Dataset):
    def __init__(self, dataset_name="uoft-cs/cifar10", split='train'):
        """
        Args:
            split (str): Split of the dataset to load ('train' or 'test').
            transform (callable, optional): A function/transform to apply to the images.
        """
        if split == "train":
            self.transform = T.Compose([
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        self.dataset = load_dataset(dataset_name, split=split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            A tuple (image, label) where image is a transformed PIL image and label is the class index.
        """
        sample = self.dataset[idx]
        image = sample['img'].convert('RGB') # Convert to RGB
        label = sample['label']

        if self.transform:
            image = self.transform(image)

        return {"x": image}, label
