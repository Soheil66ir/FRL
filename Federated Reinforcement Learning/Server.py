import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.server import ServerConfig
import torch
import os
import glob


# Import your model class here
from Networks import Actor, Critic

# Initialise Network's model (without loading weights yet)
actor = Actor(state_dim=25, goal_dim=3, action_dim=4, max_action=1.0)
critic = Critic(state_dim=25, goal_dim=3, action_dim=4)

# Metric Aggregation functions
def aggregate_success_rate(metrics):
    success_rates = [m[1]["success_rate"] for m in metrics]
    return sum(success_rates) / len(success_rates)

def aggregate_average_reward(metrics):
    average_rewards = [m[1]["average_reward"] for m in metrics]
    return sum(average_rewards) / len(average_rewards)

def fit_metrics_aggregation_fn(fit_metrics):
    return {
        "success_rate": aggregate_success_rate(fit_metrics),
        "average_reward": aggregate_average_reward(fit_metrics),
    }

def evaluate_metrics_aggregation_fn(evaluate_metrics):
    return {
        "success_rate": aggregate_success_rate(evaluate_metrics),
        "average_reward": aggregate_average_reward(evaluate_metrics),
    }

# Define a custom strategy to save and load model checkpoints
class SaveModelStrategy(FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
    ):
        # Aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            # Convert Parameters to List[np.ndarray]
            aggregated_ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Convert List[np.ndarray] to PyTorch state_dict
            actor_dict = {k: torch.tensor(v) for k, v in zip(actor.state_dict().keys(), aggregated_ndarrays[:len(actor.state_dict())])}
            critic_dict = {k: torch.tensor(v) for k, v in zip(critic.state_dict().keys(), aggregated_ndarrays[len(actor.state_dict()):])}

            # Save the model state_dicts
            torch.save({'actor': actor_dict, 'critic': critic_dict}, f"model_round_{server_round}.pth")

        return aggregated_parameters, aggregated_metrics

# Load the latest checkpoint based on time of save, if available
def load_latest_checkpoint():
    list_of_files = glob.glob("model_round_*.pth")
    if not list_of_files:
        print("No pre-trained model found, starting with new model.")
        return None

    latest_round_file = max(list_of_files, key=os.path.getctime)
    print("Loading pre-trained model from:", latest_round_file)

    checkpoint = torch.load(latest_round_file)
    actor.load_state_dict(checkpoint['actor'])
    critic.load_state_dict(checkpoint['critic'])

    # Convert to Parameters for Flower
    actor_params = [v.cpu().numpy() for v in actor.state_dict().values()]
    critic_params = [v.cpu().numpy() for v in critic.state_dict().values()]
    return fl.common.ndarrays_to_parameters(actor_params + critic_params)

# Load the latest model parameters if available
initial_parameters = load_latest_checkpoint()

# Define the strategy with the custom save model behavior and load pre-trained model if available
strategy = SaveModelStrategy(
    min_available_clients=3, # Minimum number of clients
    min_fit_clients=3, 
    min_evaluate_clients=3,
    initial_parameters=initial_parameters,
    fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
    evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn
)

# Define server config
config = ServerConfig(num_rounds=2) # Number of training rounds

if __name__ == "__main__":
    # Start Flower server with the custom strategy
    fl.server.start_server(
        server_address="localhost:8080",
        config=config,
        strategy=strategy,
    )
