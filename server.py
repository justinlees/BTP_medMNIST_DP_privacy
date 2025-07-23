# server.py (Modified)

from multiprocessing import Pool
from client import client_update, evaluate
from model import CNN
from utils import average_weights
from dataset import get_global_test_loader
import torch
import matplotlib.pyplot as plt # Import matplotlib

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Plotting Function (can be moved to a separate plot_results.py if preferred) ---
def plot_results(global_accuracies, client_accuracies, client_training_times, num_rounds):
    rounds = range(1, num_rounds + 1)

    # Plot Accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(rounds, global_accuracies, label='Global Model Accuracy', marker='o', linestyle='-', color='blue')
    for client_id, accuracies in client_accuracies.items():
        # Ensure client_accuracies lists match the number of rounds
        if len(accuracies) == num_rounds:
            plt.plot(rounds, accuracies, label=f'Client {client_id} Accuracy', linestyle='--', alpha=0.7)
        else:
            print(f"Warning: Client {client_id} accuracy history length ({len(accuracies)}) does not match num_rounds ({num_rounds}). Skipping client accuracy plot for this client.")

    plt.title('Accuracy per Round (Global vs. Clients)')
    plt.xlabel('Communication Round')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.xticks(rounds)
    plt.tight_layout()
    plt.show()

    # Plot Training Time
    plt.figure(figsize=(12, 6))
    for client_id, times in client_training_times.items():
        if len(times) == num_rounds:
            plt.plot(rounds, times, label=f'Client {client_id} Training Time', marker='x', linestyle='-.', alpha=0.7)
        else:
            print(f"Warning: Client {client_id} training time history length ({len(times)}) does not match num_rounds ({num_rounds}). Skipping client training time plot for this client.")

    plt.title('Training Time per Round (Each Client)')
    plt.xlabel('Communication Round')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    plt.legend()
    plt.xticks(rounds)
    plt.tight_layout()
    plt.show()


def simulate_federated_learning(num_clients, num_rounds, learning_rate, dp_config, privacy_method):
    global_model = CNN().to(DEVICE)
    global_weights = global_model.state_dict()

    global_test_loader = get_global_test_loader()

    # --- Metric Storage ---
    global_accuracies_history = []
    client_local_accuracies_history = {cid: [] for cid in range(num_clients)}
    client_training_times_history = {cid: [] for cid in range(num_clients)}
    # --- End Metric Storage ---

    print("\n--- Starting Federated Learning Simulation ---")
    for rnd in range(num_rounds):
        print(f"\n--- Round {rnd+1}/{num_rounds} ---")
        args = [(i, global_weights, learning_rate, dp_config, privacy_method, num_clients) for i in range(num_clients)]

        with Pool(processes=num_clients) as pool:
            # client_update now returns (state_dict, local_accuracy, training_duration)
            round_results = pool.starmap(client_update, args)

        # Separate results and populate histories
        current_round_client_weights = []
        for cid, (weights, local_acc, train_time) in enumerate(round_results):
            current_round_client_weights.append(weights)
            client_local_accuracies_history[cid].append(local_acc)
            client_training_times_history[cid].append(train_time)

        global_weights = average_weights(current_round_client_weights)
        global_model.load_state_dict(global_weights)

        print(f"\n[Server] Evaluating global model on round {rnd+1}...")
        global_loss, global_accuracy = evaluate(global_model, global_test_loader)
        global_accuracies_history.append(global_accuracy) # Store global accuracy
        print(f"[Server] Global evaluation round {rnd+1}: Loss {global_loss:.4f}, Accuracy {global_accuracy:.4f}")

    print("\n--- Training completed ---")
    final_loss, final_accuracy = evaluate(global_model, global_test_loader)
    print(f"\n--- Final Global Model Performance ---")
    print(f"Final Loss: {final_loss:.4f}")
    print(f"Final Accuracy: {final_accuracy:.4f}")

    # --- Plot results at the end ---
    plot_results(global_accuracies_history, client_local_accuracies_history, client_training_times_history, num_rounds)
    # --- End Plotting ---