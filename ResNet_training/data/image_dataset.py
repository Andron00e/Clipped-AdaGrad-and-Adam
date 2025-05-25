import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image as PILImage
import torchvision.transforms as T

class HFImageDataset(Dataset):
    def __init__(self, dataset_name="uoft-cs/cifar10", split='train'):
        """
        Initializes the image dataset with specified transformations and loads the dataset.
        Args:
            dataset_name (str, optional): The name or path of the dataset to load. Defaults to "uoft-cs/cifar10".
            split (str, optional): The dataset split to use, either 'train' or another split (e.g., 'test'). Defaults to 'train'.
        Attributes:
            transform (torchvision.transforms.Compose): The transformation pipeline applied to the images.
                - For 'train' split: Applies random cropping, random horizontal flipping, tensor conversion, and normalization.
                - For other splits: Applies tensor conversion and normalization only.
            dataset (datasets.Dataset): The loaded dataset object from the Hugging Face datasets library.
            random_ids (None): Placeholder for random indices, initialized as None.
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
        print(f"Dataset of len {len(self.dataset)}")
        self.random_ids = None
        

    def __len__(self):
        if not self.random_ids is None:
            return len(self.random_ids)
        else:
            return len(self.dataset)
    
    def random_crop(self, ids):
        self.random_ids = ids

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            A tuple (image, label) where image is a transformed PIL image and label is the class index.
        """
        # whether random_crop is applied
        if not self.random_ids is None:
            idx = self.random_ids[idx]
        
        sample = self.dataset[idx]
        image = sample['img'].convert('RGB') # Convert to RGB
        label = sample['label']

        if self.transform:
            image = self.transform(image)

        return {"x": image}, label
