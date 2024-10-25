from pair import pair
import asyncio
import pandas as pd
from dotdict import DotDict as dotdict
import wandb
import copy

WANDB = True

async def main():
    if WANDB:
        # wandb.init(project="jailbreak-llms", entity="jailbreak-llms")
        pass # TODO: logger init here etc etc. 

    args = dotdict()

    # const args 
    args.attack_model = "gpt-4o"
    args.target_model = "claude-3-haiku-20240307"
    args.judge_model = "gpt-4o-mini"

    args.attack_max_n_tokens = 1000
    args.max_n_attack_attempts = 5
    args.target_max_n_tokens = 250
    args.judge_max_n_tokens = 100
    args.judge_temperature = 0

    args.n_streams = 5
    args.keep_last_n = 3
    args.n_iterations = 5

    args.wandb = WANDB


    data = pd.read_csv("data/harmful_behaviors_custom.csv")

    tasks = []
    
    # var args
    data = data.head(3)
    for i, row in data.iterrows():
        args.index = row["Original index"]
        args.goal = row["goal"]
        args.target_str = row["target"]
        args.category = row["category"]

        await pair(args)
    #     tasks.append(pair(args))
    # await asyncio.gather(*tasks)



    # asyncio.run(pair(args))

if __name__ == "__main__":
    asyncio.run(main())