from correctingagent.experiments import experiment
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name", type=str)
    parser.add_argument("colour_model_config", type=str)

    args = parser.parse_args()
    experiment.add_big_experiment(args.experiment_name, args.colour_model_config)

