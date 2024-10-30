from pair import pair
import asyncio
import pandas as pd
from dotdict import DotDict as dotdict
import wandb
import copy
from loggers import WandBLogger

from parse_log import do_prints

async def main():
    args = dotdict()

    TESTING = False 

    # const args 
    args.attack_model = "gpt-4o-mini" if TESTING else "gpt-4o"
    args.target_model = "claude-3-haiku-20240307" if TESTING else "claude-3-sonnet-20240229"
    args.judge_model = "gpt-4o-mini" if TESTING else "gpt-4o"

    args.wandb = True
    args.quiet = not TESTING
    args.reply = True

    args.attack_max_n_tokens = 1000
    args.max_n_attack_attempts = 5
    args.target_max_n_tokens = 250
    args.judge_max_n_tokens = 100
    args.judge_temperature = 0

    args.n_streams = 2 if TESTING else 20  # number of parallel threads
    args.keep_last_n = 3
    args.n_iterations = 3  # max depth of search

    if args.wandb:
        proj_name = "reply-pair" if args.reply else "baseline-pair"
        if not TESTING: proj_name += "-full" 
        name = None if TESTING else f"{args.attack_model}-vs-{args.target_model}"
        args.logger = WandBLogger(args, project_name = proj_name, entity = "arunim_a", name = name)
    data = pd.read_csv("data/harmful_behaviors_custom.csv")

    if TESTING: data = data.head(3)

    # tasks = []
    full_out = []
    # var args
    for i, row in data.iterrows():
        args.index = row["Original index"]
        args.goal = row["goal"]
        args.target_str = row["target"]
        args.category = row["category"]

        out, q2jb = await pair(args)
        full_out.append((args.index, out, q2jb))
    #     tasks.append(pair(args))
    # full_out = await asyncio.gather(*tasks)
    print(f"<main.py_output>{full_out}</main.py_output>")

    do_prints(full_out, args)

    if args.wandb:
        args.logger.logger.finish()


if __name__ == "__main__":
    asyncio.run(main())
    #TODO: parse args instead of hardcoding