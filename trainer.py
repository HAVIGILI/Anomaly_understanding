import torch
import torch.nn as nn
import torch.optim as optim

class Trainer:
    def __init__(self, model, train_loader, test_loader, num_epochs, learning_rate=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs

        # Define the loss function
        self.criterion = nn.CrossEntropyLoss()

        # Define the optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()  # Set model to training mode
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(self.train_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)  # [batch_size, num_classes]
                
                # Compute loss
                loss = self.criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Accumulate loss
                running_loss += loss.item()
                
                if (i + 1) % 100 == 0:
                    print(f"Epoch [{epoch+1}/{self.num_epochs}], Step [{i+1}/{len(self.train_loader)}], Loss: {running_loss / 100:.4f}")
                    running_loss = 0.0
            
            # Validation after each epoch
            self.validate(epoch)

    def validate(self, epoch):
        self.model.eval()  # Set model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)  # [batch_size, num_classes]
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{self.num_epochs}], Accuracy on test set: {accuracy:.2f}%")
