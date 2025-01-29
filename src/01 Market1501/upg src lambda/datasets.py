import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

class Market1501Dataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file {csv_file} not found.")
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Image directory {img_dir} not found.")
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        required_columns = ['filename', 'id', 'age', 'backpack', 'bag', 'handbag', 'downblack', 
                            'downblue', 'downbrown', 'downgray', 'downgreen', 'downpink', 'downpurple', 
                            'downwhite', 'downyellow', 'upblack', 'upblue', 'upgreen', 'upgray', 
                            'uppurple', 'upred', 'upwhite', 'upyellow', 'clothes', 'down', 'up', 
                            'hair', 'hat', 'gender']

        for column in required_columns:
            if column not in self.annotations.columns:
                raise ValueError(f"Required column '{column}' is missing from the annotations CSV file.")

        logger.info(f"Successfully loaded annotations from {csv_file}. Found {len(self.annotations)} samples.")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        try:
            img_path = os.path.join(self.img_dir, self.annotations.iloc[idx]['filename'].strip())
            image = Image.open(img_path).convert('RGB')

            if self.transform:
                image = self.transform(image)

            labels = {
                'reid': torch.tensor(self.annotations.iloc[idx]['id'] - 1, dtype=torch.long),
                'age': torch.tensor(self.annotations.iloc[idx]['age'] - 1, dtype=torch.long),
                'backpack': torch.tensor(self.annotations.iloc[idx]['backpack'] - 1, dtype=torch.long),
                'bag': torch.tensor(self.annotations.iloc[idx]['bag'] - 1, dtype=torch.long),
                'handbag': torch.tensor(self.annotations.iloc[idx]['handbag'] - 1, dtype=torch.long),
                'clothes': torch.tensor(self.annotations.iloc[idx]['clothes'] - 1, dtype=torch.long),
                'down': torch.tensor(self.annotations.iloc[idx]['down'] - 1, dtype=torch.long),
                'up': torch.tensor(self.annotations.iloc[idx]['up'] - 1, dtype=torch.long),
                'hair': torch.tensor(self.annotations.iloc[idx]['hair'] - 1, dtype=torch.long),
                'hat': torch.tensor(self.annotations.iloc[idx]['hat'] - 1, dtype=torch.long),
                'gender': torch.tensor(self.annotations.iloc[idx]['gender'] - 1, dtype=torch.long),
                'downblack': torch.tensor(self.annotations.iloc[idx]['downblack'] - 1, dtype=torch.long),
                'downblue': torch.tensor(self.annotations.iloc[idx]['downblue'] - 1, dtype=torch.long),
                'downbrown': torch.tensor(self.annotations.iloc[idx]['downbrown'] - 1, dtype=torch.long),
                'downgray': torch.tensor(self.annotations.iloc[idx]['downgray'] - 1, dtype=torch.long),
                'downgreen': torch.tensor(self.annotations.iloc[idx]['downgreen'] - 1, dtype=torch.long),
                'downpink': torch.tensor(self.annotations.iloc[idx]['downpink'] - 1, dtype=torch.long),
                'downpurple': torch.tensor(self.annotations.iloc[idx]['downpurple'] - 1, dtype=torch.long),
                'downwhite': torch.tensor(self.annotations.iloc[idx]['downwhite'] - 1, dtype=torch.long),
                'downyellow': torch.tensor(self.annotations.iloc[idx]['downyellow'] - 1, dtype=torch.long),
                'upblack': torch.tensor(self.annotations.iloc[idx]['upblack'] - 1, dtype=torch.long),
                'upblue': torch.tensor(self.annotations.iloc[idx]['upblue'] - 1, dtype=torch.long),
                'upgreen': torch.tensor(self.annotations.iloc[idx]['upgreen'] - 1, dtype=torch.long),
                'upgray': torch.tensor(self.annotations.iloc[idx]['upgray'] - 1, dtype=torch.long),
                'uppurple': torch.tensor(self.annotations.iloc[idx]['uppurple'] - 1, dtype=torch.long),
                'upred': torch.tensor(self.annotations.iloc[idx]['upred'] - 1, dtype=torch.long),
                'upwhite': torch.tensor(self.annotations.iloc[idx]['upwhite'] - 1, dtype=torch.long),
                'upyellow': torch.tensor(self.annotations.iloc[idx]['upyellow'] - 1, dtype=torch.long),
            }

            return {'image': image, **labels}

        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            raise


def get_data_loaders(train_csv, train_img_dir, batch_size=32, num_workers=4, shuffle=True, pin_memory=True, config=None):
    if config and 'transform' in config:
        transform = config['transform']
    else:
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ])

    dataset = Market1501Dataset(train_csv, train_img_dir, transform=transform)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    logger.info(f"Created DataLoader for training with {len(data_loader)} batches.")
    return data_loader
