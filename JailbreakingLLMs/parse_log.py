import re
from dotdict import DotDict as dotdict

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

def main(file_path):
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

        do_prints(data)


if __name__ == "__main__":
    # fpath = "/Users/arunim/Documents/github/reply/JailbreakingLLMs/wandb/latest-run"
    baseline_pair_full_path = "/Users/arunim/Documents/github/reply/JailbreakingLLMs/wandb/run-20241025_222841-pm3y0vic"
    reply_pair_full_path = "/Users/arunim/Documents/github/reply/JailbreakingLLMs/wandb/run-20241029_222932-lukeuhyc"
    print("Baseline PAIR Full")
    main(baseline_pair_full_path)
    print("Reply PAIR Full")
    main(reply_pair_full_path)