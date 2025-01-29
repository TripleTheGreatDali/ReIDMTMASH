import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import logging

from model import MultiTaskReIDModel
from datasets import get_data_loader


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def evaluate_attribute(predictions, labels, is_binary=True):
    if is_binary:
        preds = torch.sigmoid(predictions).round().squeeze()
    else:
        preds = predictions.argmax(dim=1)
    accuracy = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
    return accuracy

def test_model(test_loader, model, device):
    model.eval()
    all_metrics = {}

    binary_attributes = ['gender', 'hat', 'boots', 'backpack', 'bag', 'handbag']
    multiclass_attributes = ['shoes', 'top', 'downblack', 'downwhite', 'downred', 'downblue', 'downbrown',
                             'upblack', 'upwhite', 'upred', 'upblue', 'upbrown']

    with torch.no_grad():
        for data in test_loader:
            inputs = data['image'].to(device)
            outputs = model(inputs)
            for task, output in outputs.items():
                if task != 'reid' and task in data:
                    labels = data[task].to(device)
                    is_binary = task in binary_attributes
                    is_multiclass = task in multiclass_attributes

                    if is_binary or is_multiclass:
                        accuracy = evaluate_attribute(output, labels, is_binary)
                        if task not in all_metrics:
                            all_metrics[task] = []
                        all_metrics[task].append(accuracy)

    avg_metrics = {task: sum(values) / len(values) for task, values in all_metrics.items()}
    return avg_metrics

def main():
    test_csv = './Processing/duke_attribute_train.csv'
    test_img_dir = './Data/archive/bounding_box_train'
    model_checkpoint_path = './path/to/save/models/last_trained_model.pth'
    
    backbone_type = 'resnet50'
    embed_dim = 256
    num_classes = 63
    batch_size = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_loader = get_data_loader(
        test_csv,
        test_img_dir,
        batch_size=batch_size,
        num_workers=1,
        shuffle=False,
        pin_memory=True
    )

    model = MultiTaskReIDModel(
        backbone_type=backbone_type,
        embed_dim=embed_dim,
        num_classes=num_classes,
        pretrained=True
    ).to(device)
    
    try:
        checkpoint = torch.load(model_checkpoint_path,weights_only=True, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
    except Exception as e:
        logger.error(f"Failed to load model weights: {e}")
        return


    metrics = test_model(test_loader, model, device)
    for task, accuracy in metrics.items():
        logger.info(f"Accuracy for {task}: {accuracy:.4f}")

if __name__ == '__main__':
    main()
