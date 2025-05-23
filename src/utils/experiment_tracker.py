import json
import datetime

def log_experiment(config, metrics, model_path):
    experiment_log = {
        "timestamp": datetime.datetime.now().isoformat(),
        "config": config,
        "metrics": metrics,
        "model_path": model_path
    }
    
    with open("experiments.json", "a") as f:
        f.write(json.dumps(experiment_log) + "\n")
