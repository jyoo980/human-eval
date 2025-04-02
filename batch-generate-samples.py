#!/usr/bin/env python3

import glob
import subprocess

task_files = glob.glob("*-prompt.jsonl")
generate_task_cmds = [
    f"python3 generate-samples-for-task.py --task {f}" for f in task_files
]

for cmd in generate_task_cmds:
    print(f"Executing '{cmd}'")
    subprocess.run(cmd, shell=True, check=True, capture_output=True, encoding="utf-8")
