import re
from dotdict import DotDict as dotdict
import matplotlib.pyplot as plt
import numpy as np

# IGNORE OG_INDEX, it's meaningless here. 

def do_prints(full_out, args = None):
    if args is None:
        args = dotdict()
        args.n_streams = 20
        args.n_iterations = 3

    # count how many of these have max_score number of queries = None
    num_failed = sum(1 for og_index, max_score, num_queries in full_out if num_queries is None)
    print(f"Number of failed jailbreak attempts: {num_failed} / {len(full_out)}")

    # assert that all of these have max_score = 0
    assert all(max_score == 0 for og_index, max_score, num_queries in full_out if num_queries is None), "Expected all failed jailbreak attempts to have max_score = 0"

    # average the number of queries over the non-failed attempts
    avg_queries = sum(num_queries for og_index, max_score, num_queries in full_out if num_queries is not None) / (len(full_out) - num_failed)
    print(f"Average number of queries needed to jailbreak over non-failed attempts: {avg_queries:.2f}")

    # average the number of queries over all attempts, replacing failed attempts with args.n_streams * args.n_iterations
    avg_queries = (sum(num_queries for og_index, max_score, num_queries in full_out if num_queries is not None) + num_failed * args.n_streams * args.n_iterations) / len(full_out)
    print(f"Average number of queries needed to jailbreak over all attempts: {avg_queries:.2f}")

def graph_data(data_list):

    # Set up colors and transparencies
    colors = {'Haiku': 'blue', 'Sonnet': 'red'}
    alphas = {'Baseline': 1.0, 'Reply': 0.5}

    num_datasets = len(data_list)  # Should be 4
    num_data_points = len(data_list[0][0])  # Assuming all datasets have the same length

    indices = np.arange(num_data_points)  # indices from 0 to num_data_points -1

    width = 0.2

    offsets = [(-width * (num_datasets - 1) / 2) + i * width for i in range(num_datasets)]

    # Prepare data for plotting
    queries_lists = []
    labels = []
    colors_list = []
    alphas_list = []

    max_queries = max(
        num_queries
        for data, _ in data_list
        for _, _, num_queries in data
        if num_queries is not None
    )

    min_queries = -5  # For failed attempts

    for i, (data, description) in enumerate(data_list):
        num_queries_list = []
        for _, _, num_queries in data:
            if num_queries is not None:
                num_queries_list.append(num_queries)
            else:
                num_queries_list.append(min_queries)  # Set to -5 for failed attempts
        queries_lists.append(num_queries_list)
        labels.append(description)
        # Determine color and alpha
        model = 'Haiku' if 'Haiku' in description else 'Sonnet'
        mode = 'Baseline' if 'Baseline' in description else 'Reply'
        colors_list.append(colors[model])
        alphas_list.append(alphas[mode])

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 7))

    for i in range(num_datasets):
        ax.bar(indices + offsets[i], queries_lists[i], width=width, label=labels[i],
               color=colors_list[i], alpha=alphas_list[i])

    # Plot dashed horizontal lines for averages (excluding None values)
    for i, (data, description) in enumerate(data_list):
        num_queries_non_none = [num_queries for _, _, num_queries in data if num_queries is not None]
        avg_queries = sum(num_queries_non_none) / len(num_queries_non_none)
        model = 'Haiku' if 'Haiku' in description else 'Sonnet'
        mode = 'Baseline' if 'Baseline' in description else 'Reply'
        color = colors[model]
        alpha = alphas[mode]
        linestyle = '--' 

        ax.axhline(y=avg_queries, color=color, alpha=alpha, linestyle=linestyle, linewidth=2)
        # Annotate the average line
        ax.text(num_data_points - 0.5, avg_queries + 1, f'Avg: {avg_queries:.2f}',
                color='black', fontsize=9, ha='left', va='bottom')

    # Customize y-axis to include -5 labeled as "Failed"
    y_ticks = ax.get_yticks()
    if min_queries not in y_ticks:
        y_ticks = np.append(y_ticks[y_ticks > min_queries], min_queries)
        y_ticks = np.sort(y_ticks)
    ax.set_yticks(y_ticks)
    y_tick_labels = [str(int(tick)) if tick != min_queries else 'Failed' for tick in y_ticks]
    ax.set_yticklabels(y_tick_labels)

    ax.set_xlabel("Index")
    ax.set_ylabel("Number of Queries")
    ax.set_title("Number of Queries to Jailbreak")
    ax.set_xticks(indices)
    ax.set_xticklabels([str(i) for i in indices], rotation=90)

    # Adjust legend to only include datasets
    ax.legend(title='Datasets')

    plt.tight_layout()
    plt.show()

def main(file_paths, descriptions):
    data_list = []
    for file_path, description in zip(file_paths, descriptions):
        file_path = file_path + "/files/output.log"
        with open(file_path, "r") as f:
            # find the text between <main.py_output> and </main.py_output>
            text = re.findall(r'<main.py_output>(.*)</main.py_output>', f.read())
            
            # check to see if there is only one match
            if len(text) != 1:
                raise ValueError("Expected exactly one match for <main.py_output> and </main.py_output>, found {}".format(len(text)))
            
            text = text[0] # this should be a string representation of a list of tuples

            # parse the string into a list of tuples
            data = eval(text)

            print(description)
            print(data)
            do_prints(data)
            data_list.append((data, description))

    graph_data(data_list)


if __name__ == "__main__":
    # fpath = "/Users/arunim/Documents/github/reply/JailbreakingLLMs/wandb/latest-run"
    haiku_baseline_full_path = "/Users/arunim/Documents/github/reply/JailbreakingLLMs/wandb/run-20241025_222841-pm3y0vic"
    haiku_reply_full_path = "/Users/arunim/Documents/github/reply/JailbreakingLLMs/wandb/run-20241029_222932-lukeuhyc"

    sonnet_baseline_full_path = "/Users/arunim/Documents/github/reply/JailbreakingLLMs/wandb/run-20241030_095028-53hpb9dh"
    sonnet_reply_full_path = "/Users/arunim/Documents/github/reply/JailbreakingLLMs/wandb/run-20241030_111551-cw3mfv10"

    data_list = [haiku_baseline_full_path, haiku_reply_full_path, sonnet_baseline_full_path, sonnet_reply_full_path]
    descriptions = ["Haiku Baseline", "Haiku Reply", "Sonnet Baseline", "Sonnet Reply"]

    main(data_list, descriptions)
