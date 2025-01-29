import os
import pandas as pd
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

class DukeMTMCDataset(Dataset):

    def __init__(self, csv_file, img_dir, transform=None):
        if not os.path.exists(csv_file):
            logger.error(f"CSV file not found at {csv_file}")
            raise FileNotFoundError(f"CSV file {csv_file} not found.")
        if not os.path.exists(img_dir):
            logger.error(f"Image directory not found at {img_dir}")
            raise FileNotFoundError(f"Image directory {img_dir} not found.")

        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform or self.default_transforms()

        required_columns = ['path', 'person_id', 'backpack', 'bag', 'handbag', 'boots', 
                            'gender', 'hat', 'shoes', 'top', 'downblack', 'downwhite', 
                            'downred', 'downgray', 'downblue', 'downgreen', 'downbrown', 
                            'upblack', 'upwhite', 'upred', 'uppurple', 'upgray', 'upblue', 
                            'upgreen', 'upbrown', 'image_index']
        missing_columns = [col for col in required_columns if col not in self.annotations.columns]
        if missing_columns:
            logger.error(f"CSV missing columns: {', '.join(missing_columns)}")
            raise ValueError(f"CSV missing columns: {', '.join(missing_columns)}")

        logger.info(f"Successfully loaded annotations from {csv_file}. Found {len(self.annotations)} samples.")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx]['path'])
        try:
            image = Image.open(img_path).convert('RGB')
        except UnidentifiedImageError as e:
            logger.error(f"Error opening image at {img_path}: {e}")
            raise

        if self.transform:
            image = self.transform(image)

        labels = {
            'reid': torch.tensor(self.annotations.iloc[idx]['person_id'] - 1, dtype=torch.long),
            'top': torch.tensor(self.annotations.iloc[idx]['top'] - 1, dtype=torch.long),
            'backpack': torch.tensor(self.annotations.iloc[idx]['backpack'] - 1, dtype=torch.long),
            'bag': torch.tensor(self.annotations.iloc[idx]['bag'] - 1, dtype=torch.long),
            'handbag': torch.tensor(self.annotations.iloc[idx]['handbag'] - 1, dtype=torch.long),
            'boots': torch.tensor(self.annotations.iloc[idx]['boots'] - 1, dtype=torch.long),
            'gender': torch.tensor(self.annotations.iloc[idx]['gender'] - 1, dtype=torch.long),
            'hat': torch.tensor(self.annotations.iloc[idx]['hat'] - 1, dtype=torch.long),
            'shoes': torch.tensor(self.annotations.iloc[idx]['shoes'] - 1, dtype=torch.long),
            'downblack': torch.tensor(self.annotations.iloc[idx]['downblack'] - 1, dtype=torch.long),
            'downwhite': torch.tensor(self.annotations.iloc[idx]['downwhite'] - 1, dtype=torch.long),
            'downred': torch.tensor(self.annotations.iloc[idx]['downred'] - 1, dtype=torch.long),
            'downgray': torch.tensor(self.annotations.iloc[idx]['downgray'] - 1, dtype=torch.long),
            'downblue': torch.tensor(self.annotations.iloc[idx]['downblue'] - 1, dtype=torch.long),
            'downgreen': torch.tensor(self.annotations.iloc[idx]['downgreen'] - 1, dtype=torch.long),
            'downbrown': torch.tensor(self.annotations.iloc[idx]['downbrown'] - 1, dtype=torch.long),
            'upblack': torch.tensor(self.annotations.iloc[idx]['upblack'] - 1, dtype=torch.long),
            'upwhite': torch.tensor(self.annotations.iloc[idx]['upwhite'] - 1, dtype=torch.long),
            'upred': torch.tensor(self.annotations.iloc[idx]['upred'] - 1, dtype=torch.long),
            'uppurple': torch.tensor(self.annotations.iloc[idx]['uppurple'] - 1, dtype=torch.long),
            'upgray': torch.tensor(self.annotations.iloc[idx]['upgray'] - 1, dtype=torch.long),
            'upblue': torch.tensor(self.annotations.iloc[idx]['upblue'] - 1, dtype=torch.long),
            'upgreen': torch.tensor(self.annotations.iloc[idx]['upgreen'] - 1, dtype=torch.long),
            'upbrown': torch.tensor(self.annotations.iloc[idx]['upbrown'] - 1, dtype=torch.long)
        }

        return {'image': image, **labels}

    @staticmethod
    def default_transforms():
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def get_data_loader(csv_file, img_dir, batch_size=32, shuffle=True, num_workers=4, pin_memory=True):
    dataset = DukeMTMCDataset(csv_file, img_dir)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    logger.info(f"Data loader created with {len(dataset)} samples.")
    return data_loader

if __name__ == "__main__":
    loader = get_data_loader('', '', batch_size=16)
    for batch in loader:
        print(batch['image'].shape)
        for key, value in batch.items():
            if key != 'image':
                print(f"{key}: {value}")
