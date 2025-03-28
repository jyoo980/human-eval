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


TASK_DIRECTORY_PATTERN = r"humaneval\d"
# Much simpler to use the code below once we have the full set of tasks
# NUM_HUMANEVAL_TASKS = 164
# TASK_DIRS = [f"humaneval{i}" for i in range(NUM_HUMANEVAL_TASKS)]
TASK_DIRS = [
    directory
    for directory in glob.glob("humaneval*")
    if re.match(TASK_DIRECTORY_PATTERN, directory)
]

TAGS = ["DESCRIPTION", "INPUTS", "OUTPUT", "EXAMPLES"]
TAG_PERMUTATIONS = list(permutations(TAGS))
PERMUTATION_FILE_PREFIXES = ["-".join(perm) for perm in TAG_PERMUTATIONS]


def _run_correctness_check(task_directory: str):
    """Runs OpenAI's HumanEval test harness for each of the samples in the given
    task directory. The end result is a `.jsonl` file for each sample file. Each
    line in the result file corresponds to a single sample and whether it
    passed/failed its corresponding test suite.

    Args:
        task_directory (str): The directory containing the sample files for
        which to run the HumanEval test harness.
    """
    sample_files = [
        # Sample file names are currently in this ugly format ending in
        # '-prompt.jsonl-samples.jsonl'. generate-samples.py has been updated to
        # avoid this format in the future. For now, we need this for our
        # experiments.
        f"{permutation}-prompt.jsonl-samples.jsonl"
        for permutation in PERMUTATION_FILE_PREFIXES
    ]
    solution_files = [f"{perm}-solution.jsonl" for perm in PERMUTATION_FILE_PREFIXES]
    for sample_file, solution_file in zip(sample_files, solution_files):
        evaluate_functional_correctness(
            sample_file=f"{task_directory}/{sample_file}",
            problem_file=f"{task_directory}/{solution_file}",
        )

for task_dir in TASK_DIRS:
    _run_correctness_check(task_dir)
