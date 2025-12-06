import os, argparse, json
from datetime import datetime
import numpy as np
import pandas as pd

from src.data_utils import load_uci_heart, make_targets
from src.evaluate import evaluate_models

def create_experiment_dir(base_dir: str, task: str, cv_folds: int, seed: int, 
                          data_source: str, models: str) -> str:
    """Create a unique experiment directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"exp_{task}_cv{cv_folds}_seed{seed}_{timestamp}"
    exp_dir = os.path.join(base_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir

def save_experiment_metadata(exp_dir: str, args, data_meta: dict):
    """Save experiment parameters and metadata to JSON."""
    metadata = {
        "experiment_settings": {
            "task": args.task,
            "cv_folds": args.cv_folds,
            "seed": args.seed,
            "models": args.models,
            "data_source": args.data_source,
            "local_path": args.local_path,
        },
        "data_metadata": data_meta,
        "timestamp": datetime.now().isoformat(),
        "command": f"python run_experiments.py --task {args.task} --cv-folds {args.cv_folds} --seed {args.seed} --data-source {args.data_source}" + 
                   (f" --models {args.models}" if args.models != 'all' else "") +
                   (f" --local-path {args.local_path}" if args.local_path else "")
    }
    
    metadata_path = os.path.join(exp_dir, "experiment_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    return metadata_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='binary', choices=['binary','multiclass'])
    parser.add_argument('--cv-folds', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--models', type=str, default='all', help="comma-separated subset or 'all'")
    parser.add_argument('--data-source', type=str, default='ucimlrepo', choices=['ucimlrepo','local'])
    parser.add_argument('--local-path', type=str, default=None, help='path to local CSV if using --data-source local')
    parser.add_argument('--out-dir', type=str, default='outputs', help='base output directory (experiments will be in subfolders)')
    args = parser.parse_args()

    X, y_raw, meta = load_uci_heart(source=args.data_source, local_path=args.local_path)
    y = make_targets(y_raw, task=args.task).values

    models_to_run = None if args.models=='all' else [m.strip() for m in args.models.split(',')]

    # Create unique experiment directory
    os.makedirs(args.out_dir, exist_ok=True)
    exp_dir = create_experiment_dir(args.out_dir, args.task, args.cv_folds, args.seed, 
                                    args.data_source, args.models)
    
    # Save experiment metadata
    metadata_path = save_experiment_metadata(exp_dir, args, meta)
    print(f"=== Experiment Directory: {exp_dir} ===")
    print(f"=== Metadata saved to: {metadata_path} ===")
    
    result = evaluate_models(X, y, task=args.task, cv_folds=args.cv_folds,
                             seed=args.seed, models_to_run=models_to_run, out_dir=exp_dir)

    print("\n=== Finished. Key outputs ===")
    for k,v in result.items():
        print(f"{k}: {v}")

if __name__ == '__main__':
    main()
