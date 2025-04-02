#!/usr/bin/env python3

import glob
import subprocess
from datetime import datetime


def _get_folder_name(full_path_to_permutation_folder: str) -> str:
    _, task_folder = full_path_to_permutation_folder.split("/")
    _, number_part, _ = task_folder.split("-")
    return f"humaneval{number_part}"


# task_permutation_folders = glob.glob("humaneval-task-permutations/*")

for folder in ["task-permutations-humaneval/humaneval-0-permutations"]:
    print(f"Processing {folder}")
    folder_name = _get_folder_name(folder)
    create_folder_cmd = f"mkdir {folder_name}"
    cp_permutation_prompts = "cp " + f"{folder}/* {folder_name}/"
    cp_scripts_cmd = (
        "cp ./{generate-samples-for-task.py,batch-generate-samples.py} " + folder_name
    )
    run_batch_generate_cmd = f"cd {folder_name} && ./batch-generate-samples.py"
    go_back_up_cmd = "cd ../"

    # Create the folder
    subprocess.run(
        create_folder_cmd, shell=True, check=True, capture_output=True, encoding="utf-8"
    )
    # Copy scripts
    subprocess.run(
        cp_scripts_cmd, shell=True, check=True, capture_output=True, encoding="utf-8"
    )
    # Copy permutation files
    subprocess.run(
        cp_permutation_prompts,
        shell=True,
        check=True,
        capture_output=True,
        encoding="utf-8",
    )
    # cd and run batch-generate-samples.py
    generation_start_time = datetime.now()
    print(
        f"[{generation_start_time.strftime('%Y-%m-%d %H:%M')}] Generating code for: {folder}"
    )
    subprocess.run(
        run_batch_generate_cmd,
        shell=True,
        check=True,
        capture_output=True,
        encoding="utf-8",
    )
    generation_end_time = datetime.now()
    print(
        f"[{generation_end_time.strftime('%Y-%m-%d %H:%M')}] Finished generating code for: {folder}"
    )
    print(
        f"Time elapsed (sec): {(generation_end_time - generation_start_time).total_seconds()}"
    )
    # cd back up
    subprocess.run(
        go_back_up_cmd, shell=True, check=True, capture_output=True, encoding="utf-8"
    )
