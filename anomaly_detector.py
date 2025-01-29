import torch
import torchvision
from torch.utils.data import DataLoader
import pytorch_ood
from pytorch_ood.utils import OODMetrics
from pytorch_ood.detector import Mahalanobis, MultiMahalanobis, OpenMax  # Ensure this import exists

class AnomalyDetector:
    def __init__(self, model, id_fit_data, id_test_data, ood_test_data, device):
        self.device = device
        self.model = model.to(self.device).eval()
        self.fit_loader = DataLoader(id_fit_data, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(ood_test_data + id_test_data, batch_size=32, shuffle=True)

    def openmax(self):
        detector = OpenMax(self.model)
        metrics = OODMetrics
        detector.fit(self.test_loader, device = self.device)

        for x, y in self.test_loader:
            metrics.update(detector(x).to(self.device), y)

        print(metrics.compute()) 

    def mahalanobis(self):
        detector = Mahalanobis(self.model)  # Initialize properly
        metrics = OODMetrics()

        for x, y in self.test_loader:
            metrics.update(detector(x).to(self.device), y)

        print(metrics.compute())

    def multimahalanobis(self, layers, plot_scores=False):
        detector = MultiMahalanobis(layers)  # Initialize properly
        detector.fit(self.fit_loader, device=self.device)  # Use self.device

        metrics = OODMetrics()

        for x, y in self.test_loader:
            metrics.update(detector(x).to(self.device), y)

        print(metrics.compute())
