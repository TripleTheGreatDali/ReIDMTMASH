import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class GeMPooling(nn.Module):
    def __init__(self, p=3.0, eps=1e-6, trainable=True):
        super(GeMPooling, self).__init__()
        if trainable:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        return self.gem(x, self.p, self.eps)

    @staticmethod
    def gem(x, p, eps):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)

    def __repr__(self):
        p = self.p.data.item() if isinstance(self.p, nn.Parameter) else self.p
        return f"{self.__class__.__name__}(p={p:.4f}, eps={self.eps})"

class Backbone(nn.Module):
    def __init__(self, backbone_type='efficientnet', pretrained=True):
        super(Backbone, self).__init__()
        if backbone_type == 'resnet50':
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            self.features = nn.Sequential(*list(resnet.children())[:-2])
            self.gem_pooling = GeMPooling()
            self.output_dim = resnet.fc.in_features
        elif backbone_type == 'efficientnet':
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
            self.features = efficientnet.features
            self.gem_pooling = GeMPooling()
            self.output_dim = efficientnet.classifier[1].in_features
        else:
            raise NotImplementedError(f"Backbone type '{backbone_type}' is not implemented.")

    def forward(self, x):
        x = self.features(x)
        x = self.gem_pooling(x)
        x = x.view(x.size(0), -1)
        return x

class ReIDHead(nn.Module):
    def __init__(self, input_dim=1280, embed_dim=256, num_classes=63):
        super(ReIDHead, self).__init__()
        self.fc = nn.Linear(input_dim, embed_dim)
        self.norm = nn.BatchNorm1d(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.fc(x)
        x = self.norm(x)
        x = F.normalize(x, p=2, dim=1)
        logits = self.classifier(x)
        return logits

class BinaryAttributeHead(nn.Module):
    def __init__(self, input_dim):
        super(BinaryAttributeHead, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        logits = self.fc(x)
        return logits

class MultiClassAttributeHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MultiClassAttributeHead, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        logits = self.fc(x)
        return logits

class MultiTaskReIDModel(nn.Module):
    def __init__(self, backbone_type='efficientnet', embed_dim=256, num_classes=63, pretrained=True):
        super(MultiTaskReIDModel, self).__init__()
        self.backbone = Backbone(backbone_type=backbone_type, pretrained=pretrained)
        input_dim = self.backbone.output_dim

        self.reid_head = ReIDHead(input_dim=input_dim, embed_dim=embed_dim, num_classes=num_classes)
        self.gender_head = BinaryAttributeHead(input_dim=input_dim)
        self.top_head = BinaryAttributeHead(input_dim=input_dim)
        self.boots_head = BinaryAttributeHead(input_dim=input_dim)
        self.hat_head = BinaryAttributeHead(input_dim=input_dim)
        self.backpack_head = BinaryAttributeHead(input_dim=input_dim)
        self.bag_head = BinaryAttributeHead(input_dim=input_dim)
        self.handbag_head = BinaryAttributeHead(input_dim=input_dim)
        self.shoes_color_head = MultiClassAttributeHead(input_dim=input_dim, num_classes=3)

        self.upper_color_heads = nn.ModuleList([MultiClassAttributeHead(input_dim=input_dim, num_classes=8) for _ in range(8)])
        self.lower_color_heads = nn.ModuleList([MultiClassAttributeHead(input_dim=input_dim, num_classes=7) for _ in range(7)])

        self.tasks = ['reid', 'gender', 'top', 'boots', 'hat', 'backpack', 'bag', 'handbag', 'shoes_color', 'upper_colors', 'lower_colors']

    def forward(self, x):
        shared_features = self.backbone(x)
        outputs = {
            'reid': self.reid_head(shared_features),
            'gender': self.gender_head(shared_features),
            'top': self.top_head(shared_features),
            'boots': self.boots_head(shared_features),
            'hat': self.hat_head(shared_features),
            'backpack': self.backpack_head(shared_features),
            'bag': self.bag_head(shared_features),
            'handbag': self.handbag_head(shared_features),
            'shoes_color': self.shoes_color_head(shared_features),
            'upper_colors': [head(shared_features) for head in self.upper_color_heads],
            'lower_colors': [head(shared_features) for head in self.lower_color_heads]
        }
        return outputs

if __name__ == "__main__":
    model = MultiTaskReIDModel(
        backbone_type='efficientnet',
        embed_dim=256,
        num_classes=63,
        pretrained=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    input_tensor = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
    print("Output keys:", list(outputs.keys()))

    try:
        from torchsummary import summary
        summary(model, input_size=(3, 224, 224), device=str(device))
    except ImportError:
        print("torchsummary is not installed. Printing model architecture:")
        print(model)
