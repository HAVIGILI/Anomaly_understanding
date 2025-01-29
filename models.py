import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
import pytorch_ood
import pytorch_ood.model as model

class ModelCreator:
    def __init__(self, model_name = "WideResNet", num_classes = 10):
        model.WideResNet
        try:
            self.model = getattr(model, model_name)(num_classes = num_classes, pretrained = "cifar10-pt")
            print("hello")
        except AttributeError:
            print("model name", model_name, "not found, using default WideResNet")
            self.model = model.WideResNet(num_classes = num_classes, pretrained = "cifar10-pt")

def main():
    creator = ModelCreator()
    print(creator.model)
        
if __name__ == "__main__":
    main()


        