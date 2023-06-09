"""
An example script to generate datasets for different speeds in parallel
"""
import os
import numpy as np
from multiprocessing import Process, Queue
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--sionna", default=False, action="store_true")
parser.add_argument("--start_speed", type=int, default=0)
parser.add_argument("--end_speed", type=int, default=200)
parser.add_argument("--step", type=int, default=5)
parser.add_argument("--size", type=int, default=10000)
parser.add_argument("--scenario", type=str, default="umi")
parser.add_argument("--pilot_pattern", type=str, default="block")
parser.add_argument("--p_spacing", type=int, default=1)
parser.add_argument("--symbol_indices", nargs="+", type=int, default=[3, 10])
parser.add_argument("--nr", type=int, default=4, help="num receive antennas")
parser.add_argument("--nu", type=int, default=2, help="num transmit antennas")
parser.add_argument(
    "--num-processes",
    type=int,
    default=1,
    help="number of generation to be run in parallel",
)
parser.add_argument("--save_dir", type=str, default="./datasets")
args = parser.parse_args()

commands = []
save_dir = args.save_dir

use_sionna = args.sionna

ue_speeds = np.arange(args.start_speed, args.end_speed + args.start_speed, args.step)

main_path = f"{args.save_dir}/{args.scenario}_{args.pilot_pattern}"

pilot_symbols = "--pilot_ofdm_symbol_indices "

for idx in args.symbol_indices:
    pilot_symbols += f"{idx} "

for ue_speed in ue_speeds:
    if use_sionna:
        cmd = (
            "python scripts/generate_datasets_from_sionna.py "
            f"--output_dir {main_path}/ "
            "--seed 0 "
            f"--size {args.size} "
            f"--scenario {args.scenario} "
            "--batch_size 100 "
            f"--num_rx_antennas {args.nr} "
            f"--n_ues {args.nu} "
            "--path_loss "
            f"--p_spacing {args.p_spacing} --pilot_pattern {args.pilot_pattern} "
            f"{pilot_symbols}"
            f"--ue_speed {ue_speed} "
        )
        print(cmd)
        commands.append(cmd)
    else:
        raise ValueError("Please specify the link level simulator to use")


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
    task_queue.put(cmd)

# Submit stop signals

for i in range(args.num_processes):
    task_queue.put("STOP")

# Start worker processes

for i in range(args.num_processes):
    Process(target=worker, args=(task_queue, done_queue)).start()

# Get and print results

for i in range(args.num_processes):
    print(f"Process {i}", done_queue.get())
