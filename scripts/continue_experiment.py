from correctingagent.experiments import experiment
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_id", type=int)

    args = parser.parse_args()

    experiment.continue_big_experiment(args.experiment_id)


