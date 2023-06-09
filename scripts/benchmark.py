import os, sys
import numpy as np
import random
from multiprocessing import Process, Queue
import argparse

from cebed.models import MODELS

INPUT_TYPES = {
    "ReEsNet": "low",
    "InReEsNet": "low",
    "MReEsNet": "raw",
    "DDAE": "raw",
    "HA02": "low",
    "MTRE": "raw",
    "ChannelNet": "low",
}

parser = argparse.ArgumentParser(description="Channel Estimation Benchmark")
########################## General args ###################################
parser.add_argument("--experiment_name", type=str, required=True)
parser.add_argument("--data_dir", type=str)

parser.add_argument("--dataset_name", type=str, default="SionnaOfflineMD")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("-v", "--verbose", type=int, default=1)

parser.add_argument("--output_dir", type=str, default="train_output")
########################## Training params ################################
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--train_batch_size", type=int, default=32)
parser.add_argument("--eval_batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--loss_fn", default="mse", type=str, help="The loss function")
parser.add_argument("--train_split", type=float, default=0.9)
parser.add_argument(
    "--early_stopping", action="store_true", help="If early stopping should be used"
)
parser.add_argument(
    "--input_type",
    type=str,
    help="The input type for data preprocessing",
    default=None,
    choices=["low", "raw"],
)
parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2])

args = parser.parse_args()

commands = []

for i in range(len(MODELS)):
    gid = i % len(args.gpus)
    print(gid)
    cmd = (
        f" CUDA_VISIBLE_DEVICES={gid} python scripts/train.py --experiment_name {args.experiment_name} --seed {args.seed} --output_dir {args.output_dir} "
        f"--data_dir {args.data_dir} -v {args.verbose} --lr {args.lr} "
        f"--train_batch_size {args.train_batch_size} --eval_batch_size {args.eval_batch_size} --epochs {args.epochs} "
        f"--dataset_name {args.dataset_name} --model_name {MODELS[i]} --input_type {INPUT_TYPES[MODELS[i]]}"
    )
    commands.append(cmd)


def worker(input, output):
    for cmd in iter(input.get, "STOP"):
        ret_code = os.system(cmd)

        if ret_code != 0:
            output.put("killed")

            break
    output.put("done")


# Create queues
task_queue = Queue()
done_queue = Queue()

# Submit tasks

for cmd in commands:
    print(cmd)
    task_queue.put(cmd)

# Submit stop signals
num_processes = len(args.gpus)
for i in range(num_processes):
    task_queue.put("STOP")

# Start worker processes

for i in range(num_processes):
    Process(target=worker, args=(task_queue, done_queue)).start()

# Get and print results

for i in range(num_processes):
    print(f"Process {i}", done_queue.get())
