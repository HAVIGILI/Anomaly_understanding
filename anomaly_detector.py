from torch.utils.data import DataLoader
from pytorch_ood.utils import OODMetrics
from pytorch_ood.detector import Mahalanobis, MultiMahalanobis, OpenMax
import matplotlib.pyplot as plt

class AnomalyDetector:
    def __init__(self, model, id_fit_data, id_test_data, ood_test_data, device):
        self.device = device
        self.model = model.to(self.device).eval()
        self.fit_loader = DataLoader(id_fit_data, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(ood_test_data + id_test_data, batch_size=32, shuffle=True)

    def _test_and_plot_scores(self, detector, metrics, plot_scores=False):
        """
        Helper function

        """
        id_scores = []
        ood_scores = []

        for x, y in self.test_loader:
            x = x.to(self.device)
            scores = detector(x)
            metrics.update(scores, y)

            if plot_scores:
                for score, label in zip(scores, y):
                    if label == -1:
                        ood_scores.append(score.item())
                    else:
                        id_scores.append(score.item())

        if plot_scores:
            plt.hist(id_scores, bins=50, alpha=0.5, label='ID')
            plt.hist(ood_scores, bins=50, alpha=0.5, label='OOD')
            plt.legend(loc='upper right')
            plt.title("OOD Scores Distribution")
            plt.xlabel("OOD Score")
            plt.ylabel("Frequency")
            plt.show()

        print(metrics.compute())

    def openmax(self, plot_scores):
        detector = OpenMax(self.model)
        metrics = OODMetrics()
        detector.fit(self.test_loader, device=self.device)

        # Use the helper function
        self._test_and_plot_scores(detector, metrics, plot_scores=plot_scores)

    def mahalanobis(self, plot_scores = False):
        detector = Mahalanobis(self.model)
        metrics = OODMetrics()
        detector.fit(self.fit_loader, device=self.device)

        self._test_and_plot_scores(detector, metrics, plot_scores=plot_scores)

    def multimahalanobis(self, layers, plot_scores=False):
        detector = MultiMahalanobis(layers)
        metrics = OODMetrics()
        detector.fit(self.fit_loader, device=self.device)

        self._test_and_plot_scores(detector, metrics, plot_scores=plot_scores)
