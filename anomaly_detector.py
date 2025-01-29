import torch
import torchvision
from torch.utils.data import DataLoader
import pytorch_ood
from pytorch_ood.utils import OODMetrics
from pytorch_ood.detector import Mahalanobis, MultiMahalanobis, OpenMax 
import matplotlib.pyplot as plt

class AnomalyDetector:
    def __init__(self, model, id_fit_data, id_test_data, ood_test_data, device):
        self.device = device
        self.model = model.to(self.device).eval()
        self.fit_loader = DataLoader(id_fit_data, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(ood_test_data + id_test_data, batch_size=32, shuffle=True)

    def openmax(self):
        detector = OpenMax(self.model)
        metrics = OODMetrics()
        detector.fit(self.test_loader, device=self.device)

        for x, y in self.test_loader:
            x = x.to(self.device)  
            metrics.update(detector(x), y)

        print(metrics.compute()) 

    def mahalanobis(self):
        detector = Mahalanobis(self.model)
        metrics = OODMetrics()
        detector.fit(self.fit_loader, device=self.device)

        for x, y in self.test_loader:
            x = x.to(self.device)
            metrics.update(detector(x), y)

        print(metrics.compute())

    def multimahalanobis(self, layers, plot_scores=False):
        detector = MultiMahalanobis(layers)
        detector.fit(self.fit_loader, device=self.device)

        metrics = OODMetrics()

        for x, y in self.test_loader:
            x = x.to(self.device)
            metrics.update(detector(x), y)

        print(metrics.compute())

        id_scores = []
        ood_scores = []

        z=0
        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.device)
                scores = detector(x)
                for score, label in zip(scores, y):
                    z+=1
                    if label == -1:
                        ood_scores.append(score.item())
                    else:
                        id_scores.append(score.item())
        print(z)
        plt.hist(id_scores, bins=50, alpha=0.5, label='ID')
        plt.hist(ood_scores, bins=50, alpha=0.5, label='OOD')
        plt.legend(loc='upper right')
        plt.title("OOD Scores Distribution")
        plt.xlabel("OOD Score")
        plt.ylabel("Frequency")
        plt.show()
