import glob
import re

from itertools import permutations
from pathlib import Path

# This script automatically validates whether the expected set of sample files
# for tag permutations have been generated for each of the tasks in the
# HumanEval dataset.
#
# human-eval/
# ├─ humaneval1/
# │  ├─ INPUT-OUTPUT-DESCRIPTION-EXAMPLES-samples.jsonl
# │  ├─ INPUT-OUTPUT-DESCRIPTION-EXAMPLES-solution.jsonl
# ├─ humaneval2/
# ├─ check-generate-samples-result.py
#
# Example usage:
# python run-correctness-checks.py


TASK_DIRECTORY_PATTERN = r"humaneval\d"
TASK_DIRS = [
    directory
    for directory in glob.glob("humaneval*")
    if re.match(TASK_DIRECTORY_PATTERN, directory)
]

NUMBER_OF_HUMANEVAL_TASKS = 164
TASKS_WITHOUT_EXAMPLES = {38, 41, 83}

TAGS = ["DESCRIPTION", "INPUTS", "OUTPUT", "EXAMPLES"]
TAG_PERMUTATIONS = list(permutations(TAGS))
PERMUTATION_FILE_PREFIXES = ["-".join(perm) for perm in TAG_PERMUTATIONS]
EXPECTED_SAMPLE_FILES = [
    f"{perm}-prompt.jsonl-samples.jsonl" for perm in PERMUTATION_FILE_PREFIXES
]


def _all_task_dirs_exist() -> bool:
    """Checks whether all task directories (i.e., 'humaneval0' to 'humaneval163)
    exist immediately under the directory in which this script is executed.

    Returns:
        bool: True if all task directories resulting from generating samples for
        the HumanEval dataset exist immediately under the directory in which this
        script is executed.
    """
    task_dir_numbers = [int(re.findall(r"\d+", task_dir)[0]) for task_dir in TASK_DIRS]
    task_dir_numbers.sort()
    return task_dir_numbers == list(range(NUMBER_OF_HUMANEVAL_TASKS))


def _all_samples_present(task_directory: str) -> bool:
    """Checks whether all sample files for a given task have been successfully
    generated.

    All task directories (with the exception of those corresponding to the tasks
    in the set `TASKS_WITHOUT_EXAMPLES`) should contain 24 sample files, each
    corresponding to the number of permutations possible for the set `TAGS`.

    The missing sample files in the directories corresponding to the tasks in
    the set `TASKS_WITHOUT_EXAMPLES` should all correspond to permutations that
    include the `EXAMPLES` tag.

    Args:
        task_directory (str): The directory for which to check the presence of
        sample files.

    Returns:
        bool: True if all expected sample files are present in the given
        directory.
    """
    sample_file_paths = [f"{task_directory}/{path}" for path in EXPECTED_SAMPLE_FILES]
    sample_files_to_existence = {
        path: Path(path).exists() for path in sample_file_paths
    }
    if not all(sample_files_to_existence.values()):
        task_number = int(re.findall(r"\d+", task_directory)[0])
        if task_number not in TASKS_WITHOUT_EXAMPLES:
            return False
        else:
            missing_sample_files_for_task = [
                sample_file
                for sample_file, does_file_exist in sample_files_to_existence.items()
                if not does_file_exist
            ]
            return all(
                "EXAMPLES" in missing_sample_file
                for missing_sample_file in missing_sample_files_for_task
            )


assert _all_task_dirs_exist()
for task_dir in TASK_DIRS:
    assert _all_samples_present(task_dir)
