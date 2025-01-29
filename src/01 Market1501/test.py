import os
import torch
import logging
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from model import MultiTaskReIDModel
from datasets import get_data_loaders


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

TEST_CSV = './new_test.csv'
TEST_IMG_DIR = './data/bounding_box_test'
MODEL_PATH = './checkpoints/best_model.pth'
BATCH_SIZE = 16 

def compute_accuracy(preds, labels, task_type):
    if task_type == 'binary':
        preds = (torch.sigmoid(preds) > 0.5).long()
    elif task_type == 'multiclass':
        preds = preds.argmax(dim=1)
    return accuracy_score(labels.cpu(), preds.cpu())

def test_model(test_loader, model, device):
    model.eval()
    metrics = {
        'gender_accuracy': [], 'hair_accuracy': [], 'up_accuracy': [], 'down_accuracy': [], 'hat_accuracy': [],
        'backpack_accuracy': [], 'bag_accuracy': [], 'handbag_accuracy': [],
        'clothes_accuracy': [], 'age_accuracy': []
    }

    with torch.no_grad():
        for data in tqdm(test_loader, desc="Testing Model", leave=False):
            inputs = data['image'].to(device)
            labels_binary = {task: data[task].to(device) for task in ['gender', 'hair', 'up', 'down', 'hat', 'backpack', 'bag', 'handbag']}
            labels_multiclass = {task: data[task].to(device) for task in ['clothes', 'age']}

            outputs = model(inputs)


            for task, label in labels_binary.items():
                acc = compute_accuracy(outputs[task], label, 'binary')
                metrics[f'{task}_accuracy'].append(acc)


            for task, label in labels_multiclass.items():
                acc = compute_accuracy(outputs[task], label, 'multiclass')
                metrics[f'{task}_accuracy'].append(acc)

    summary_metrics = {key: sum(vals) / len(vals) for key, vals in metrics.items()}
    logger.info("Test Summary Metrics: {}".format(summary_metrics))

    metrics_df = pd.DataFrame([summary_metrics])
    results_path = 'test_metrics.csv'
    metrics_df.to_csv(results_path, index=False)
    logger.info(f"Test metrics saved to {results_path}")

    return summary_metrics

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_loader = get_data_loaders(
        TEST_CSV,
        TEST_IMG_DIR,
        batch_size=BATCH_SIZE,
        num_workers=4,
        shuffle=False,
        pin_memory=True
    )

    model = MultiTaskReIDModel(
        backbone_type='resnet50',
        embed_dim=256,
        num_age_classes=4,
        num_clothes_classes=3,
        pretrained=False
    ).to(device)

    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True), strict=False)
    model.eval()

    test_metrics = test_model(test_loader, model, device)
    logger.info("Final Test Summary Metrics: {}".format(test_metrics))

if __name__ == '__main__':
    main()
