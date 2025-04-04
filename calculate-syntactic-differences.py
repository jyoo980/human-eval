import glob
import re
import json

from itertools import permutations, combinations
from tsed.TSED import Calculate

# This script computes the average TSED (Tree Similarity of Edit Distance)
# score across the samples generated for each tag permutation for a task.
#
# The following directory structure is assumed by this script:
#
# human-eval/
# ├─ humaneval1/
# │  ├─ INPUT-OUTPUT-DESCRIPTION-EXAMPLES-samples.jsonl
# │  ├─ INPUT-OUTPUT-DESCRIPTION-EXAMPLES-solution.jsonl
# ├─ humaneval2/
# ├─ calculate-syntactic-differences.py
#
# Example usage:
# python calculate-syntactic-differences.py


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


def _average_score_for_sample(samples: list[str]) -> float:
    """Calculates the average TSED score across a list of samples
    (i.e., code completions) generated for a given task.

    A total TSED score for the entire set of samples is calculated by
    pairing each sample with the rest in the list. This total score
    is divided by the number of combinations to yield an average TSED
    score.

    Args:
        samples (list[str]): A list of samples, which are code completions.

    Returns:
        float: The TSED score calculated for the list of samples.
    """
    if not samples:
        return 0
    # TODO: should this be a combination or a permutation? I.e., calculating
    # TSED involves providing an origin and target AST, and the number of edits
    # differs depending on what is passed.
    pairs_of_samples = list(combinations(samples, r=2))
    total_tsed_score = 0
    for origin, target in pairs_of_samples:
        # TODO: What are sensible weights for deletion, insertion, and rename operations?
        score_for_pair = Calculate(
            "python",
            origin,
            target,
            deletion_weight=1.0,
            insertion_weight=1.0,
            rename_weight=0.5,
        )
        total_tsed_score += score_for_pair
    return total_tsed_score / len(pairs_of_samples)


def _run_syntactic_similarity_checks(task_directory: str):
    """Runs syntactic similarity checks for all of the samples
    contained within the given directory.

    Args:
        task_directory (str): The name of a directory that contains
            samples for a programming task.
    """
    sample_files = [
        # Sample file names are currently in this ugly format ending in
        # '-prompt.jsonl-samples.jsonl'. generate-samples.py has been updated to
        # avoid this format in the future. For now, we need this for our
        # experiments.
        f"{task_directory}/{permutation}-samples.jsonl"
        for permutation in PERMUTATION_FILE_PREFIXES
    ]
    for sample_file in sample_files:
        code_generated_for_sample = []
        with open(sample_file, "r") as f:
            samples_in_file = f.readlines()
            code_generated_for_sample = [
                json.loads(sample)["completion"] for sample in samples_in_file
            ]
            _average_score_for_sample(code_generated_for_sample)
        # TODO: report the score, either to a file or the console.


for task_dir in TASK_DIRS:
    _run_syntactic_similarity_checks(task_dir)
