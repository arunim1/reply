# %%
import matplotlib.pyplot as plt
import json
import numpy as np

# %%
def plot_loss(json_paths, name=None, baseline=None, alpha=0.3):
    plt.figure(figsize=(10, 6))
    
    # Plot baseline if provided
    if baseline is not None:
        if isinstance(baseline, str):
            baseline = [baseline]    
        baseline_losses = []
        for path in baseline:
            with open(path, 'r') as f:
                data = json.load(f)
            losses = data['losses']
            baseline_losses.extend(losses)
        baseline_losses = np.array(baseline_losses)
        plt.plot(baseline_losses, 'k--', label='Baseline', alpha=0.8)

    # Plot actual runs
    if isinstance(json_paths, str):
        json_paths = [json_paths]
    
    all_losses = []
    for path in json_paths:
        with open(path, 'r') as f:
            data = json.load(f)
        losses = data['losses']
        all_losses.extend(losses)
    
    plt.plot(np.array(all_losses), 'r-', label='Reply mode')
    
    plt.xlabel('GCG Step')
    plt.ylabel('Loss')
    if name is not None:
        plt.title(name)
    else:
        plt.title('Loss vs GCG Step')
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

def main():
    # Example usage:
    # Plot baseline with multiple runs overlaid
    plot_loss([
            'v2_outputs/testing2_run3.json',
            'v2_outputs/testing2_run2.json',
            'v2_outputs/testing2_run1.json',
            'v2_outputs/testing2_run4.json',
            'v2_outputs/testing2_run5.json',
        ],
        name='Testing Runs',
        baseline='v2_outputs/testing2_baseline.json')

    plot_loss([
            'v3_outputs/testing3_run3.json',
            'v3_outputs/testing3_run2.json',
            'v3_outputs/testing3_run1.json',
            'v3_outputs/testing3_run4.json',
            'v3_outputs/testing3_run5.json',
        ],
        name='Testing3 Runs',
        baseline='v3_outputs/testing3_baseline.json')

if __name__ == "__main__":
    main()