# main.py (No significant changes from last step, keeping for clarity)
import argparse
from server import simulate_federated_learning

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--privacy", choices=["none", "dp", "secure_agg"], default="none")
    args = parser.parse_args()

    dp_config = None
    learning_rate = 0.001 # Default learning rate for non-DP

    if args.privacy == "dp":
        dp_config = {
            'clip_grad_norm': 1.0,
            'noise_multiplier': 0.1,
            'target_delta': 1e-5,
            'epochs': 5 # Pass local epochs to dp_config if make_private needs it
        }
        learning_rate = 0.0005 # Very low learning rate for DP training
        print(f"DP ENABLED: Learning Rate set to {learning_rate}, DP Config: {dp_config}")
    elif args.privacy == "none":
        print(f"DP DISABLED: Learning Rate set to {learning_rate}")
    elif args.privacy == "secure_agg":
        # For secure_agg, you might still want to use a specific learning rate
        # and potentially some 'noise_std' parameter for your custom implementation
        print(f"Secure Aggregation (custom noise) ENABLED: Using default LR {learning_rate}")
        # If your secure_agg method needed config, you'd define it here too
        pass # No specific dp_config for this custom secure_agg noise
    else:
        print(f"Privacy mode '{args.privacy}' not handled. Using default LR.")


    simulate_federated_learning(
        num_clients=args.clients,
        num_rounds=args.rounds,
        learning_rate=learning_rate,
        dp_config=dp_config,
        privacy_method=args.privacy # Pass the privacy method string
    )