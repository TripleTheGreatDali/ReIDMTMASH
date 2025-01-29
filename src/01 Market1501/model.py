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
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            efficientnet = efficientnet_b0(weights=weights)

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
    def __init__(self, input_dim=1280, embed_dim=256, num_classes=1501):
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
    def __init__(self, input_dim=1280):
        super(BinaryAttributeHead, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        logits = self.fc(x)
        return logits

class MultiClassAttributeHead(nn.Module):
    def __init__(self, input_dim=1280, num_classes=4):
        super(MultiClassAttributeHead, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        logits = self.fc(x)
        return logits

class MultiTaskReIDModel(nn.Module):
    def __init__(self, backbone_type='efficientnet', embed_dim=256, num_classes=1501,
                 num_age_classes=4, num_clothes_classes=2, pretrained=True):
        super(MultiTaskReIDModel, self).__init__()
        self.backbone = Backbone(backbone_type=backbone_type, pretrained=pretrained)
        input_dim = self.backbone.output_dim

        self.reid_head = ReIDHead(input_dim=input_dim, embed_dim=embed_dim, num_classes=num_classes)

        # Attribute heads
        self.gender_head = BinaryAttributeHead(input_dim=input_dim)
        self.hair_head = BinaryAttributeHead(input_dim=input_dim)
        self.up_head = BinaryAttributeHead(input_dim=input_dim)
        self.down_head = BinaryAttributeHead(input_dim=input_dim)
        self.clothes_head = MultiClassAttributeHead(input_dim=input_dim, num_classes=num_clothes_classes)
        self.age_head = MultiClassAttributeHead(input_dim=input_dim, num_classes=num_age_classes)
        self.hat_head = BinaryAttributeHead(input_dim=input_dim)
        self.backpack_head = BinaryAttributeHead(input_dim=input_dim)
        self.bag_head = BinaryAttributeHead(input_dim=input_dim)
        self.handbag_head = BinaryAttributeHead(input_dim=input_dim)

        # Color heads
        self.upper_color_heads = nn.ModuleList([BinaryAttributeHead(input_dim=input_dim) for _ in range(8)])
        self.lower_color_heads = nn.ModuleList([BinaryAttributeHead(input_dim=input_dim) for _ in range(9)])

    def forward(self, x):
        # Shared features from the backbone
        shared_features = self.backbone(x)

        # ReID output
        reid_output = self.reid_head(shared_features)

        # Attribute outputs
        gender_output = self.gender_head(shared_features)
        hair_output = self.hair_head(shared_features)
        up_output = self.up_head(shared_features)
        down_output = self.down_head(shared_features)
        clothes_output = self.clothes_head(shared_features)
        age_output = self.age_head(shared_features)
        hat_output = self.hat_head(shared_features)
        backpack_output = self.backpack_head(shared_features)
        bag_output = self.bag_head(shared_features)
        handbag_output = self.handbag_head(shared_features)

        # Color outputs
        upper_color_outputs = [head(shared_features) for head in self.upper_color_heads]
        lower_color_outputs = [head(shared_features) for head in self.lower_color_heads]

        return {
            'reid': reid_output,
            'gender': gender_output,
            'hair': hair_output,
            'up': up_output,
            'down': down_output,
            'clothes': clothes_output,
            'age': age_output,
            'hat': hat_output,
            'backpack': backpack_output,
            'bag': bag_output,
            'handbag': handbag_output,
            'upper_colors': upper_color_outputs,
            'lower_colors': lower_color_outputs
        }

if __name__ == '__main__':

    model = MultiTaskReIDModel(
        backbone_type='resnet50',
        embed_dim=256,
        num_classes=1501,
        num_age_classes=4,
        num_clothes_classes=2,
        pretrained=True
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    model.eval()

    input_tensor = torch.randn(1, 3, 224, 224).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)

    print("Output keys:", outputs.keys())

    try:
        from torchsummary import summary
        summary(model, input_size=(3, 224, 224), device=str(device))
    except ImportError:
        print("torchsummary is not installed. Printing model architecture:")
        print(model)
