import pandas as pd
import wandb
from tqdm import tqdm

def fetch_data(key, entity, project):
    wandb.login(key=key)

    # Instantiate the wandb API
    api = wandb.Api(timeout=600)

    # Fetch all runs in the project
    runs = api.runs(f"{entity}/{project}")

    # Create empty DataFrames to store the loss and accuracy metrics, and configurations
    all_loss = pd.DataFrame()
    all_accuracy = pd.DataFrame()
    all_config = pd.DataFrame()

    run_data = []
    # Loop through the fetched runs
    for run in tqdm(runs):

        # Get the run's metrics
        metrics = run.history()

        # Add the run ID to the metrics DataFrame
        metrics["run_id"] = run.id

        # Append the metrics DataFrame to the run_data list
        run_data.append(metrics)


        # Create a DataFrame with the run's configuration details
        config_df = pd.DataFrame(run.config, index=[run.id])

        # Add the run's configuration DataFrame to the combined configuration DataFrame
        all_config = pd.concat([all_config, config_df])

    # Concatenate all the metrics DataFrames in the run_data list
    all_metrics_df = pd.concat(run_data, ignore_index=True)

    all_metrics_df = all_metrics_df.join(all_config, on="run_id")
    all_metrics_df.to_csv(f"{project}.csv", index=False)


if __name__ == "__main__":


    # Set the API key, entity and project name
    import argparse

    parser = argparse.ArgumentParser(description='Fetch data from W&B API')
    parser.add_argument('--key', type=str, help='W&B API key')
    parser.add_argument('--entity', type=str, help='W&B entity')
    parser.add_argument('--project', type=str, help='W&B project name')

    args = parser.parse_args()

    key = args.key
    entity = args.entity
    project = args.project

    # Call the fetch_data function with the specified key, entity and project
    fetch_data(key, entity, project)


# Path: figures/plot_data.py
# python fetch_data.py --key=<your_W&B_API_key> --entity=<your_W&B_entity> --project=<your_W&B_project_name>
