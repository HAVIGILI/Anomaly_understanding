#dneoindeindenido
from torchvision import transforms
from torch.utils.data import DataLoader

class CustomDataLoader:
    def __init__(self, train_dataset, validation_dataset):
        train_transform = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        test_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.cifar_train_loader = DataLoader(train_dataset, transform=train_transform, batch_size=32, shuffle=True)
        self.cifar_test_loader = DataLoader(validation_dataset, transform=test_transform, batch_size=32, shuffle=False)

    def get_loaders(self):
        return self.cifar_train_loader, self.cifar_test_loader

def main():
    datamaker = CustomDataLoader()
    train_data, test_data = datamaker.get_loaders()
    print(train_data)
    print(test_data)

if __name__ == "__main__":
    main()

