#dneoindeindenido
import numpy as np

class DataCreator:
    def __init__(self, dataset):
        self.data = dataset
        self.classes = np.unique(self.data)

    def print_data_structure(self):
        print("det här är storleken", len(self.data))
        print("det här är hela grejen", self.data)
        print("These are the classes", self.classes)

    def remove_data_classes(self, class_to_remove):
        self.data = self.data[self.data != class_to_remove]

def main():
    dataset = np.array([1, 2, 3, 3, 3, 2, 1, 2, 3, 1, 1], dtype = object)
    datamaker = DataCreator(dataset)
    datamaker.print_data_structure()
    datamaker.remove_data_classes(2)
    datamaker.print_data_structure()

if __name__ == "__main__":
    main()

