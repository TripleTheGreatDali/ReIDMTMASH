import os
import argparse
import logging
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pandas as pd

from model import MultiTaskReIDModel
from datasets import get_data_loader
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def compute_metrics(outputs, labels):
    metrics = {}
    for task, output in outputs.items():
        true_labels = labels[task].cpu().int()
        if output.shape[1] == 1:
            preds = torch.sigmoid(output).cpu().round().int()
        else:
            preds = output.cpu().argmax(dim=1)
        accuracy = accuracy_score(true_labels.numpy(), preds.numpy())
        precision = precision_score(true_labels.numpy(), preds.numpy(), average='macro', zero_division=0)
        recall = recall_score(true_labels.numpy(), preds.numpy(), average='macro', zero_division=0)
        f1 = f1_score(true_labels.numpy(), preds.numpy(), average='macro', zero_division=0)
        metrics[task] = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
    return metrics

def train_epoch(train_loader, model, criterion, optimizer, device, epoch, config):
    model.train()
    running_loss = 0.0
    reid_correct = 0
    reid_total = 0
    task_metrics = []

    lambda_binary = config.get('lambda_binary', 0.5)
    lambda_multi = config.get('lambda_multi', 1.0)
    lambda_color = config.get('lambda_color', 0.5)

    for data in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        inputs = data['image'].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        total_loss = 0

        for task, output in outputs.items():
            if task in data:
                labels = data[task].to(device)
                if output.shape[1] == 1:
                    labels = labels.view(-1, 1)
                    loss = criterion[task](output, labels.float())
                    total_loss += lambda_binary * loss

                elif task in ['reid', 'top', 'shoes', 'shoes_color']:
                    loss = criterion[task](output, labels)
                    total_loss += lambda_multi * loss

                elif task.startswith('up') or task.startswith('down'):
                    loss = criterion[task](output, labels)
                    total_loss += lambda_color * loss

                if task == 'reid':
                    reid_correct += (output.argmax(1) == labels).sum().item()
                    reid_total += labels.size(0)

                task_metrics.append(compute_metrics({task: output}, {task: labels}))

        total_loss.backward()
        optimizer.step()
        running_loss += total_loss.item()

    avg_loss = running_loss / len(train_loader)
    reid_accuracy = reid_correct / reid_total if reid_total > 0 else 0.0

    avg_metrics = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        avg_metrics[metric] = sum([m[task][metric] for m in task_metrics for task in m]) / len(task_metrics)

    logger.info(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f} - ReID Accuracy: {reid_accuracy:.4f} "
                f"- Average Accuracy: {avg_metrics['accuracy']:.4f} - Precision: {avg_metrics['precision']:.4f} "
                f"- Recall: {avg_metrics['recall']:.4f} - F1-score: {avg_metrics['f1']:.4f}")
    
    return avg_loss, reid_accuracy, avg_metrics


def main():
    parser = argparse.ArgumentParser(description='Train Multi-Task ReID Model')
    parser.add_argument('--config', type=str, required=True, help='Path to the JSON config file')
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    if not os.path.exists(config['save_dir']):
        os.makedirs(config['save_dir'])

    train_loader = get_data_loader(
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
        embed_dim=config['embed_dim'],
        num_classes=config['num_classes'],
        pretrained=True
    ).to(device)

    criterion = {
        'reid': nn.CrossEntropyLoss(),
        'gender': nn.BCEWithLogitsLoss(),
        'hat': nn.BCEWithLogitsLoss(),
        'boots': nn.BCEWithLogitsLoss(),
        'shoes': nn.CrossEntropyLoss(),
        'backpack': nn.BCEWithLogitsLoss(),
        'bag': nn.BCEWithLogitsLoss(),
        'handbag': nn.BCEWithLogitsLoss(),
        'top': nn.CrossEntropyLoss(),
        'shoes_color': nn.CrossEntropyLoss(),
        **{f'up{color}': nn.CrossEntropyLoss() for color in ['black', 'white', 'red', 'blue', 'brown']},
        **{f'down{color}': nn.CrossEntropyLoss() for color in ['black', 'white', 'red', 'blue', 'brown']}
    }

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])

    metrics_df = pd.DataFrame(columns=['epoch', 'train_loss', 'reid_accuracy', 'avg_accuracy', 'precision', 'recall', 'f1'])

    best_reid_accuracy = 0

    for epoch in range(config['epochs']):
        train_loss, reid_accuracy, avg_metrics = train_epoch(train_loader, model, criterion, optimizer, device, epoch, config)
        scheduler.step()

        epoch_metrics = pd.DataFrame([{
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'reid_accuracy': reid_accuracy,
            'avg_accuracy': avg_metrics['accuracy'],
            'precision': avg_metrics['precision'],
            'recall': avg_metrics['recall'],
            'f1': avg_metrics['f1']
        }])
        metrics_df = pd.concat([metrics_df, epoch_metrics], ignore_index=True)

        if reid_accuracy > best_reid_accuracy:
            best_reid_accuracy = reid_accuracy
            model_path = os.path.join(config['save_dir'], f'best_model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), model_path)
            logger.info(f"Best model saved at epoch {epoch+1}")

    last_model_path = os.path.join(config['save_dir'], 'last_trained_model.pth')
    torch.save(model.state_dict(), last_model_path)
    logger.info(f"Last trained model saved.")

    metrics_df.to_csv(os.path.join(config['save_dir'], 'training_metrics.csv'), index=False)

    logger.info("Training completed. Final metrics saved.")

if __name__ == '__main__':
    main()
