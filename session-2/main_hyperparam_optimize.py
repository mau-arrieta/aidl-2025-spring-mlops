import torch
from ray import tune
import numpy as np
from dataset import MyDataset
from model import MyModel
from utils import accuracy
import torch.optim as optim
from torch.utils.data import random_split,DataLoader

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
fixed_config = {
            "images_path" : "/Users/mauarrieta/Desktop/Catalunya UPC docs/Labs MLops/Chinese MNIST data/data/data",
            "labels_path" : "/Users/mauarrieta/Desktop/Catalunya UPC docs/Labs MLops/Chinese MNIST data/chinese_mnist.csv",
            "epochs": 3
        }

def train_single_epoch(model, train_loader, criterion, optimizer, device):
    model.train()  # Set model to training mode
    train_loss, correct, total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # Move to GPU if available

        optimizer.zero_grad()  # Clear previous gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        # Track loss & accuracy
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        
        correct_batch = (predicted == labels).sum().item()
        total_batch = labels.size(0)

        correct += correct_batch
        total += total_batch

    train_accuracy = 100 * correct / total
    avg_train_loss = train_loss / len(train_loader)
    print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
    return avg_train_loss, train_accuracy


def eval_single_epoch(model, val_loader, criterion, device):
    model.eval()  # Set model to evaluation mode
    val_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():  # No gradients needed for evaluation
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_accuracy = 100 * correct / total
    avg_val_loss = val_loss / len(val_loader)

    print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
    return avg_val_loss, val_accuracy


def custom_transform(image):
    # Convert to grayscale
    image = image.convert("L")  

    # Resize image (ensure consistent input size)
    image = image.resize((64, 64))  

    # Convert to NumPy array
    image = np.array(image, dtype=np.float32)  

    # Normalize (scale to [-1,1])
    image = (image / 255.0 - 0.5) * 2  

    # Add channel dimension (C, H, W) -> (1, 64, 64)
    image = np.expand_dims(image, axis=0)  

    # Convert to PyTorch tensor
    return torch.tensor(image, dtype=torch.float32)


def train_model(config,function):
    # Ray will populate the "config" dictionary automatically,
    # according to the defined search space.

    my_dataset = MyDataset(fixed_config["images_path"], fixed_config["labels_path"], custom_transform)
    my_model = MyModel().to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(my_model.parameters(), lr=0.001)
    
    train_size = int(fixed_config["train_ratio"] * len(my_dataset))
    val_size = len(my_dataset) - train_size
    # Perform random split
    train_dataset, val_dataset = random_split(my_dataset, [train_size, val_size])
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Train for one epoch and evaluate
    train_loss, train_accuracy = train_single_epoch(my_model, train_loader, criterion, optimizer, device)
    val_loss, val_accuracy = eval_single_epoch(my_model, val_loader, criterion, device)

    # Return the validation loss for Ray Tune optimization
    return {
        "val_loss": val_loss
    }

if __name__ == "__main__":
    # Define the device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    analysis = tune.run(
        train_model,
        metric="val_loss",
        mode="min",
        num_samples=5,
        config={
            "hyperparam_1": tune.uniform(1, 10),  # Example hyperparameter
            "hyperparam_2": tune.grid_search(["relu", "tanh"])  # Example activation function,
            
        })

    print("Best hyperparameters found were: ", analysis.best_config)
