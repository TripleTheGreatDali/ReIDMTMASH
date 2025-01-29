import os
import argparse
import logging
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pandas as pd
import numpy as np
from tqdm import tqdm

from model import MultiTaskReIDModel
from datasets import get_data_loaders

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def compute_metrics(preds, labels, task_type):
    """ Compute evaluation metrics based on the task type. """
    if task_type in ['multiclass', 'binary']:
        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average='macro', zero_division=0)
        recall = recall_score(labels, preds, average='macro', zero_division=0)
        f1 = f1_score(labels, preds, average='macro', zero_division=0)
    elif task_type == 'multilabel':
        accuracy = (preds == labels).mean()
        precision = precision_score(labels, preds, average='samples', zero_division=0)
        recall = recall_score(labels, preds, average='samples', zero_division=0)
        f1 = f1_score(labels, preds, average='samples', zero_division=0)
    return accuracy, precision, recall, f1

def train_epoch(train_loader, model, criterion_reid, criterion_binary, criterion_multiclass, criterion_color, optimizer, device, epoch, config):
    model.train()
    running_loss = 0.0
    reid_correct = 0
    reid_total = 0
    lambda_b = 0.5
    lambda_m = 1.0
    lambda_c = 0.5
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    for batch_idx, data in enumerate(tqdm(train_loader, desc=f'Training Epoch {epoch+1}', leave=False)):
        inputs = data['image'].to(device)
        labels_reid = data['reid'].to(device).long()
        labels_binary = {task: data[task].to(device).long().unsqueeze(1) for task in ['gender', 'hair', 'up', 'down', 'hat', 'backpack', 'bag', 'handbag']}
        labels_multiclass = {task: data[task].to(device).long() for task in ['clothes', 'age']}
        
        # Color labels
        upper_color_labels = [data[f'up{color}'].to(device).long().unsqueeze(1) for color in ['black', 'blue', 'green', 'gray', 'purple', 'red', 'white', 'yellow']]
        lower_color_labels = [data[f'down{color}'].to(device).long().unsqueeze(1) for color in ['black', 'blue', 'brown', 'gray', 'green', 'pink', 'purple', 'white', 'yellow']]

        optimizer.zero_grad()
        outputs = model(inputs)

        # Compute losses
        loss_reid = criterion_reid(outputs['reid'], labels_reid)
        loss_binary = sum(criterion_binary(outputs[task], labels_binary[task].float()) for task in labels_binary)*lambda_b
        loss_multiclass = sum(criterion_multiclass(outputs[task], labels_multiclass[task]) for task in labels_multiclass)*lambda_c

        # Color losses
        upper_color_loss = sum(criterion_color(outputs['upper_colors'][i], upper_color_labels[i].float()) for i in range(len(upper_color_labels)))
        lower_color_loss = sum(criterion_color(outputs['lower_colors'][i], lower_color_labels[i].float()) for i in range(len(lower_color_labels)))

        color_loss = (upper_color_loss+lower_color_loss)*lambda_c

        total_loss = loss_reid + loss_binary + loss_multiclass + color_loss
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()
        reid_correct += (outputs['reid'].argmax(1) == labels_reid).sum().item()
        reid_total += labels_reid.size(0)

        # Compute metrics for binary tasks
        for task, output in outputs.items():
            if task in labels_binary:
                predicted = (torch.sigmoid(output).squeeze() > 0.5).long().cpu().numpy()
                actual = labels_binary[task].squeeze().cpu().numpy()
                acc, prec, rec, f1 = compute_metrics(predicted, actual, 'binary')
                metrics['accuracy'].append(acc)
                metrics['precision'].append(prec)
                metrics['recall'].append(rec)
                metrics['f1'].append(f1)

    epoch_loss = running_loss / len(train_loader.dataset)
    reid_accuracy = reid_correct / reid_total
    avg_metrics = {key: np.mean(vals) for key, vals in metrics.items()}
    logger.info(f"Train Epoch {epoch+1}: Loss={epoch_loss:.4f}, ReID Accuracy={reid_accuracy:.4f}, Avg Metrics={avg_metrics}")
    return {'epoch': epoch + 1, 'epoch_loss': epoch_loss, 'reid_accuracy': reid_accuracy, **avg_metrics}


def main():
    parser = argparse.ArgumentParser(description='Train Multi-Task ReID Model with Config File')
    parser.add_argument('--config', type=str, required=True, help='Path to the JSON config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    if not os.path.exists(config['save_dir']):
        os.makedirs(config['save_dir'])

    train_loader = get_data_loaders(
        config['train_csv'],
        config['train_img_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=True,
        pin_memory=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiTaskReIDModel(
        backbone_type=config['backbone_type'],
        embed_dim=256,
        num_classes=config['num_classes'],
        num_age_classes=config['num_age_classes'],
        num_clothes_classes=config['num_clothes_classes'],
        pretrained=True
    ).to(device)

    criterion_reid = nn.CrossEntropyLoss()
    criterion_binary = nn.BCEWithLogitsLoss()
    criterion_multiclass = nn.CrossEntropyLoss()
    criterion_color = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    metrics_df = pd.DataFrame()
    best_accuracy = 0

    for epoch in range(config['epochs']):
        train_metrics = train_epoch(train_loader, model, criterion_reid, criterion_binary, criterion_multiclass,criterion_color, optimizer, device, epoch, config)
        scheduler.step()

        metrics_df = pd.concat([metrics_df, pd.DataFrame([train_metrics])], ignore_index=True)
        metrics_csv_path = os.path.join(config['save_dir'], config['log_file'])
        metrics_df.to_csv(metrics_csv_path, index=False)

        if train_metrics['reid_accuracy'] > best_accuracy:
            best_accuracy = train_metrics['reid_accuracy']
            best_model_path = os.path.join(config['save_dir'], 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)

    last_model_path = os.path.join(config['save_dir'], 'last_model.pth')
    torch.save(model.state_dict(), last_model_path)
    logger.info("Training completed. Metrics saved to {}".format(metrics_csv_path))

if __name__ == '__main__':
    main()
