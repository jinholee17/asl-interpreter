import torch
# import torch.nn as nn
# import torch.optim as optim
from tqdm import tqdm  

def train_model(model, train_loader, val_loader, num_epochs, loss_fun, optimizer, device):
    """
    training loop for model

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        num_epochs (int): Number of epochs.
        criterion: Loss function.
        optimizer: Optimizer.
        device: 'cpu' or 'cuda'.

    Returns:
        model: The trained model.
    """
    model.to(device) #depending on where training
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 20)

        # training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)

            # zero out gradients
            optimizer.zero_grad()

            # forward pass
            outputs = model(inputs)
            loss = loss_fun(outputs, labels)

            # backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_loss /= len(train_loader.dataset)
        train_accuracy = 100. * correct / total
        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}%")

        # validation 
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validating"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_fun(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader.dataset)
        val_accuracy = 100. * correct / total
        print(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.2f}%")

    return model
