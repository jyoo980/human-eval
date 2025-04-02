import glob
import re

from itertools import permutations
from human_eval.evaluation import evaluate_functional_correctness

# This script runs OpenAI's HumanEval evaluation harness across a set of sample
# files. The following directory structure is assumed by this script:
#
# human-eval/
# ├─ humaneval1/
# │  ├─ INPUT-OUTPUT-DESCRIPTION-EXAMPLES-samples.jsonl
# │  ├─ INPUT-OUTPUT-DESCRIPTION-EXAMPLES-solution.jsonl
# ├─ humaneval2/
# ├─ run-correctness-checks.py
#
# A `results.jsonl` file is produced for each pair of `samples.jsonl` and
# `solution.jsonl` files that reports the pass/fail status of unit tests in
# the evaluation harness for each sample.
# Example usage:
# python run-correctness-checks.py


def _get_task_directories() -> list[str]:
    """Returns the names of the directories containing sample and solution files
    for the HumanEval dataset.

    TODO: Re-implement this function later as
    ```python
    num_tasks = 164
    return [f"humaneval{i}" for i in range(num_tasks)]
    ```
    Once samples have been generated for all 164 tasks in the HumanEval dataset.

    Returns:
        list[str]: A list of the names of the directories containing sample and
        solution files for the HumanEval dataset.
    """
    task_directory_pattern = r"humaneval\d"
    dirs_prepended_with_humaneval = glob.glob("humaneval*")
    return [
        d for d in dirs_prepended_with_humaneval if re.match(task_directory_pattern, d)
    ]


def _get_tag_permutation_file_prefixes() -> list[str]:
    tags = ["DESCRIPTION", "INPUTS", "OUTPUT", "EXAMPLES"]
    tag_permutations = list(permutations(tags))
    return ["-".join(perm) for perm in tag_permutations]


def _run_correctness_check(task_directory: str):
    """Runs OpenAI's HumanEval test harness for each of the samples in the given
    task directory. The end result is a `.jsonl` file for each sample file. Each
    line in the result file corresponds to a single sample and whether it
    passed/failed its corresponding test suite.

    Args:
        task_directory (str): The directory containing the sample files for
        which to run the HumanEval test harness.
    """
    permutation_file_prefixes = _get_tag_permutation_file_prefixes()
    sample_files = [
        # Sample file names are currently in this ugly format ending in
        # '-prompt.jsonl-samples.jsonl'. generate-samples.py has been updated to
        # avoid this format in the future. For now, we need this for our
        # experiments.
        f"{permutation}-prompt.jsonl-samples.jsonl"
        for permutation in permutation_file_prefixes
    ]
    solution_files = [f"{perm}-solution.jsonl" for perm in permutation_file_prefixes]
    if len(sample_files) != len(solution_files):
        print(f"Unequal number of sample files and solution files for {task_directory}")
        return
    for sample_file, solution_file in zip(sample_files, solution_files):
        evaluate_functional_correctness(
            sample_file=f"{task_directory}/{sample_file}",
            problem_file=f"{task_directory}/{solution_file}",
        )


for task_dir in _get_task_directories():
    _run_correctness_check(task_dir)
