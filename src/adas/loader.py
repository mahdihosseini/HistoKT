import torch
import torch.nn as nn
from models.resnet import resnet18


class ModelLoader:

    def __init__(self, model, data=None, device=None):
        self.model = model
        self.data = data
        self.device = device

    @classmethod
    def load_from_ImageNet(cls, device=None, model=resnet18):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        if model == "ResNet18":
            model = resnet18(num_classes=1000, pretrained=True)

        return cls(model, None, device)

    @classmethod
    def load_from_path(cls, path_to_model, device=None, model="ResNet18", keep_data=False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        data = torch.load(path_to_model, map_location=device)
        if model == "ResNet18":
            model = resnet18(num_classes=data["state_dict_network"]["fc.weight"].shape[0])  # slightly janky

        try:
            model.load_state_dict(data["state_dict_network"])
        except:
            print("you dun goofed")
            raise

        return cls(model, data if keep_data else None, device)

    def replace_last_layer(self, projection_head: nn.Module, model_type="resnet18", freeze_encoder=True):
        if model_type == "ResNet18":
            if freeze_encoder:
                for param in self.model.parameters():
                    param.requires_grad = False
            self.model.fc = projection_head
        else:
            raise NotImplementedError("Unknown model type")

    def check_grad(self):
        for name, param in self.model.named_parameters():
            print(f"{name} requires grad: {param.requires_grad}")

    def get_fine_tune_model(self, num_classes, model_type="ResNet18", freeze_encoder=True):
        if model_type == "ResNet18":
            self.replace_last_layer(nn.Linear(512, num_classes), model_type=model_type, freeze_encoder=freeze_encoder)
        return self.model


def get_model(name, path, num_classes, freeze_encoder):
    if name == "ResNet18":
        if path:
            if path == "ImageNet":
                loader = ModelLoader.load_from_ImageNet(model=name)
                return loader.get_fine_tune_model(num_classes,
                                                  model_type=name,
                                                  freeze_encoder=freeze_encoder)
            else:
                loader = ModelLoader.load_from_path(path, model=name)
                return loader.get_fine_tune_model(num_classes,
                                                  model_type=name,
                                                  freeze_encoder=freeze_encoder)
    print("BAD MODEL CONFIG")
    return

