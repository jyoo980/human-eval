import glob
import re
import numpy as np
import pickle as pkl

from collections import defaultdict
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
# Two files: 'avg_pass_at_for_tasks.pkl' and 'avg_pass_at_k_scores_for_perms.pkl'
# are produced by this sccript.
#
# Example usage:
# python run-correctness-checks.py

SOLUTION_FILE_SUFFIX = "-solution.jsonl"
SAMPLE_FILE_SUFFIX = "-samples.jsonl"


def _parse_permutation(file_name: str, suffix_to_remove: str = "") -> str:
    """Returns the permutation part of a given file name (e.g.,
    'INPUT-OUTPUT-DESCRIPTION-EXAMPLES' when given
    'humaneval0/INPUT-OUTPUT-DESCRIPTION-EXAMPLES-solution.jsonl' with the
    suffix SOLUTION_FILE_PREFIX).

    Args:
        file_name (str): The file name from which to extract the permutation.
        suffix_to_remove (str, optional): The suffix to remove from the given
        file name. Defaults to "".

    Returns:
        str: The 4- (or 3-, for tasks that do not have examples in their
        docstrings) part permutation.
    """
    file_without_folder_path = re.sub(r"humaneval\d+\/", "", file_name)
    return (
        file_without_folder_path.removesuffix(suffix_to_remove)
        if suffix_to_remove
        else file_without_folder_path
    )


def _avg_pass_at_k_scores_for_tasks(
    task_to_permutations_to_pass_at_scores: dict[str, dict[str, dict[str, np.float64]]],
) -> dict[str, dict[str, np.float64]]:
    """Calculates the average pass@1 and pass@10 scores for each task in the
    HumanEval dataset.

    Args:
        task_to_permutations_to_pass_at_scores (dict[str, dict[str, dict[np.float64]]]):
        A mapping from task id (e.g., humaneval1) to another mapping from
        permutation to its pass@1 and pass@10 scores.

    Returns:
        dict[str, dict[str, np.float64]]: A mapping from task id (e.g., humaneval1)
        to its average pass@1 and pass@10 scores.
    """
    task_to_avg_pass_at_k_scores: dict[str, dict[str, np.float64]] = {}
    for (
        task,
        permutations_to_pass_at_scores,
    ) in task_to_permutations_to_pass_at_scores.items():
        pass_at_1s = []
        pass_at_10s = []
        for pass_at_keys_to_score in permutations_to_pass_at_scores.values():
            pass_at_1s.append(pass_at_keys_to_score["pass@1"])
            pass_at_10s.append(pass_at_keys_to_score["pass@10"])
        task_to_avg_pass_at_k_scores[task] = {
            "avg_pass@1": np.float64(np.mean(pass_at_1s)),
            "avg_pass@10": np.float64(np.mean(pass_at_10s)),
        }
    return task_to_avg_pass_at_k_scores


def _avg_pass_at_k_scores_for_permutations(
    task_to_permutations_to_pass_at_scores: dict[str, dict[str, dict[str, np.float64]]],
) -> dict[str, dict[str, np.float64]]:
    """Calculates the average pass@1 and pass@10 scores for each permutation of
    the tag orders in the prompts from the HumanEval dataset.

    Args:
        task_to_permutations_to_pass_at_scores (dict[str, dict[str, dict[np.float64]]]):
        A mapping from task id (e.g., humaneval1) to another mapping from
        permutation to its pass@1 and pass@10 scores.

    Returns:
        dict[str, dict[str, np.float64]]: A mapping from permutations to its
        average pass@1 and pass@10 scores across the HumanEval dataset.
    """
    permutation_to_pass_at_k_scores: dict[str, dict[str, list[np.float64]]] = (
        defaultdict(lambda: {"pass@1": [], "pass@10": []})
    )
    for (
        permutations_to_pass_at_scores
    ) in task_to_permutations_to_pass_at_scores.values():
        for permutation, pass_at_scores in permutations_to_pass_at_scores.items():
            pass_at_1, pass_at_10 = pass_at_scores["pass@1"], pass_at_scores["pass@10"]
            permutation_to_pass_at_k_scores[permutation]["pass@1"].append(pass_at_1)
            permutation_to_pass_at_k_scores[permutation]["pass@10"].append(pass_at_10)

    permutations_to_avg_pass_at_k_scores: dict[str, dict[str, np.float64]] = (
        defaultdict(lambda: {"avg_pass@1": np.float64(0), "avg_pass@10": np.float64(0)})
    )

    for permutation, pass_at_k_scores in permutation_to_pass_at_k_scores.items():
        avg_pass_at_1s, avg_pass_at_10s = (
            np.mean(pass_at_k_scores["pass@1"]),
            np.mean(pass_at_k_scores["pass@10"]),
        )
        permutations_to_avg_pass_at_k_scores[permutation]["avg_pass@1"] = np.float64(
            avg_pass_at_1s
        )
        permutations_to_avg_pass_at_k_scores[permutation]["avg_pass@10"] = np.float64(
            avg_pass_at_10s
        )

    return dict(permutations_to_avg_pass_at_k_scores)


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
    task_directory_pattern = r"humaneval\d+"
    dirs_prepended_with_humaneval = glob.glob("humaneval*")
    return [
        d for d in dirs_prepended_with_humaneval if re.match(task_directory_pattern, d)
    ]


def _get_sample_file_names(task_directory: str) -> list[str]:
    return glob.glob(f"{task_directory}/*{SAMPLE_FILE_SUFFIX}")


def _run_correctness_check(task_directory: str):
    """Runs OpenAI's HumanEval test harness for each of the samples in the given
    task directory. The end result is a `.jsonl` file for each sample file. Each
    line in the result file corresponds to a single sample and whether it
    passed/failed its corresponding test suite.

    Args:
        task_directory (str): The directory containing the sample files for
        which to run the HumanEval test harness.
    """
    sample_files = _get_sample_file_names(task_directory)
    solution_files = [
        sample_file_name.replace("samples", "solution")
        for sample_file_name in sample_files
    ]

    solution_file_name_to_score = {f: 0 for f in solution_files}
    assert len(sample_files) == len(solution_files), (
        f"Unequal number of sample files ({len(sample_files)}) to solution files ({len(solution_files)})"
    )
    for sample_file, solution_file in zip(sample_files, solution_files):
        score = evaluate_functional_correctness(
            sample_file=sample_file,
            problem_file=solution_file,
        )
        solution_file_name_to_score[solution_file] = score
    return {
        _parse_permutation(solution_file_name, SOLUTION_FILE_SUFFIX): scores
        for solution_file_name, scores in solution_file_name_to_score.items()
    }


task_to_scores_for_permutations = {}
for task_dir in _get_task_directories():
    task_to_scores_for_permutations[task_dir] = _run_correctness_check(task_dir)

avg_pass_at_k_scores_for_tasks = _avg_pass_at_k_scores_for_tasks(
    task_to_scores_for_permutations
)
avg_pass_at_k_scores_for_perms = _avg_pass_at_k_scores_for_permutations(
    task_to_scores_for_permutations
)

with open("avg_pass_at_for_tasks.pkl", "wb") as f:
    pkl.dump(avg_pass_at_k_scores_for_tasks, f, pkl.HIGHEST_PROTOCOL)

with open("avg_pass_at_for_perms.pkl", "wb") as f:
    pkl.dump(avg_pass_at_k_scores_for_perms, f, pkl.HIGHEST_PROTOCOL)
