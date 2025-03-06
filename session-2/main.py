import torch
from torch.utils.data import random_split,DataLoader
import numpy as np
from dataset import MyDataset
from model import MyModel
from utils import accuracy
import torch.optim as optim



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
config = {
        "epochs": 2,
        "train_ratio":0.7,
        "images_path" : "/Users/mauarrieta/Desktop/Catalunya UPC docs/Labs MLops/Chinese MNIST data/data/data",
        "labels_path" : "/Users/mauarrieta/Desktop/Catalunya UPC docs/Labs MLops/Chinese MNIST data/chinese_mnist.csv"
    }


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
        _, predicted = torch.max(outputs,1)
        

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




def main():
    
    my_dataset = MyDataset(config["images_path"],config["labels_path"],custom_transform)
    # Define loss function (for multi-class classification)
    criterion = torch.nn.CrossEntropyLoss()
    model = MyModel()
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    #Define train and test split
    print(f"\n\ndataset og size : {len(my_dataset)}\n\n")
    train_size = int(config["train_ratio"] * len(my_dataset))
    val_size = len(my_dataset) - train_size
    # Perform random split
    train_dataset, val_dataset = random_split(my_dataset, [train_size, val_size])
    print(f"\n\ntrain size : {len(train_dataset)}\n\n")
    print(f"\n\nvalidation size : {len(val_dataset)}\n\n")
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    for epoch in range(config["epochs"]):
        print(f"\nEpoch : {epoch}")
        train_single_epoch(model,train_loader,criterion,optimizer,device)
        eval_single_epoch(model,val_loader,criterion,device)

    return model



main()
    