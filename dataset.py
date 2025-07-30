# dataset.py (Modified)
from medmnist import PathMNIST, INFO
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms

# Define a common transform
COMMON_TRANSFORM = transforms.Compose([transforms.ToTensor()])

def _load_base_dataset(split: str, data_flag: str = 'pathmnist'):
    """Helper to load the base MedMNIST dataset split."""
    info = INFO[data_flag]
    DataClass = getattr(__import__('medmnist'), info['python_class'])
    return DataClass(split=split, transform=COMMON_TRANSFORM, download=True)

def get_client_data_loaders(client_id: int, num_clients: int, batch_size: int = 128):
    """
    Loads and partitions the PathMNIST training and validation datasets
    for a specific client.
    """
    data_flag = 'pathmnist'

    # Load full train and validation datasets
    full_train_dataset = _load_base_dataset('train', data_flag)
    full_val_dataset = _load_base_dataset('val', data_flag) # Use 'val' split for client validation

    # Partition the training data
    train_total = len(full_train_dataset)
    train_shard_size = train_total // num_clients
    train_indices = list(range(client_id * train_shard_size, (client_id + 1) * train_shard_size))
    client_train_subset = Subset(full_train_dataset, train_indices)
    train_loader = DataLoader(client_train_subset, batch_size=batch_size, shuffle=True)

    # Partition the validation data (similarly for clients, or give them a shared test set)
    # For simplicity, let's partition the 'val' set for client-side validation.
    # In some FL setups, clients might evaluate on a small local test set or not at all.
    val_total = len(full_val_dataset)
    val_shard_size = val_total // num_clients
    val_indices = list(range(client_id * val_shard_size, (client_id + 1) * val_shard_size))
    client_val_subset = Subset(full_val_dataset, val_indices)
    val_loader = DataLoader(client_val_subset, batch_size=batch_size, shuffle=False) # No need to shuffle validation

    print(f"Client {client_id}: Train samples = {len(client_train_subset)}, Val samples = {len(client_val_subset)}")
    return train_loader, val_loader

def get_global_test_loader(batch_size: int = 128):
    """
    Loads the entire PathMNIST test dataset for global model evaluation.
    """
    data_flag = 'pathmnist'
    test_dataset = _load_base_dataset('test', data_flag)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"Global Test Loader: Total samples = {len(test_dataset)}")
    return test_loader