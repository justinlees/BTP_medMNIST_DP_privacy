# client.py (Modified)
import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional
import time # Import time for measuring duration

from model import CNN
from dataset import get_client_data_loaders
from privacy import apply_privacy


print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# --- Client-side Training Function ---
def train(net, train_loader, epochs, learning_rate, privacy_method: str, dp_config: Optional[Dict]):
    """
    Trains the network on the training set, applying privacy mechanisms.
    This function calls apply_privacy to get the (potentially wrapped)
    model, optimizer, and train_loader.
    """
    net, optimizer, train_loader = apply_privacy(net, train_loader, privacy_method, learning_rate, dp_config)

    criterion = nn.CrossEntropyLoss()
    net.train()

    print(f"DEBUG (Client train): Starting {epochs} local epochs...")
    for epoch_idx in range(epochs):
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            labels = labels.squeeze().long()

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    print(f"DEBUG (Client train): Finished local training.")
    if privacy_method == "dp" and hasattr(net, '_module'):
        return net._module
    else:
        return net

# --- Client-side Evaluation Function ---
def evaluate(net, val_loader):
    """Evaluates the network on a validation set."""
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0, 0, 0
    net.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            labels = labels.squeeze().long()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    avg_loss = total_loss / total
    return avg_loss, accuracy

# --- Main Client Update Function (called by server via multiprocessing) ---
def client_update(cid: int, global_weights: Dict, learning_rate: float, dp_config: Optional[Dict], privacy_method: str, num_clients: int):
    """
    Performs local training and evaluation on a client's data.
    This function is executed by the multiprocessing pool.
    """
    net = CNN().to(DEVICE)
    net.load_state_dict(global_weights)

    train_loader, val_loader = get_client_data_loaders(cid, num_clients)

    local_epochs = 5 # Can be passed as an argument if you want to vary it

    print(f"Client {cid} starting local update with LR: {learning_rate}, Method: {privacy_method}, Epochs: {local_epochs}")

    # --- Start Timer for Training ---
    start_time = time.time()
    updated_net = train(net, train_loader, local_epochs, learning_rate, privacy_method, dp_config)
    end_time = time.time()
    training_duration = end_time - start_time
    # --- End Timer ---

    local_loss, local_accuracy = evaluate(updated_net, val_loader)
    print(f"Client {cid} local evaluation: Loss = {local_loss:.4f}, Accuracy = {local_accuracy:.4f}")

    # --- Return additional metrics ---
    return updated_net.state_dict(), local_accuracy, training_duration