from human_eval.data import write_jsonl, read_problems
from openai import OpenAI

import argparse
import sys

# This script consumes a .jsonl file in which each line is a JSON object. Each
# JSON object has a 'prompt' key that maps to a prompt for a task.
#
# It sends the prompt over to an LLM hosted via an external API and records its
# response (usually a completion to programming task) to another .jsonl file.
#
# A note on naming: The OpenAI HumanEval task harness expects specific file
# names for files that contain responses from language models (e.g.,
# -samples.jsonl).
#
# Example usage:
# python generate-samples.py --task some-task-prompt.jsonl

CODE_GENERATION_SYSTEM_PROMPT = """
Given a Python method with a docstring, generate only the completion for the
method. That is, do not repeat any of the method signature or docstring;
generate only the code that should go in the method body.
"""

DEFAULT_MODEL = "gpt-4o-mini-2024-07-18"

NUM_SAMPLES_PER_TASK = 20

# Program arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--task",
    dest="task",
    type=str,
    help="The .jsonl file comprising a code generation task",
)

llm_client = OpenAI()


def _sample_file_name(task_file_name: str) -> str:
    """Returns the name of the file that comprises the samples (i.e., the code
    generated) for a programming task. The given task file name must end with
    the string '-prompt.jsonl'.

    Args:
        task_file_name (str): The task file name, which must end with the string
        '-prompt.jsonl'

    Returns:
        str: The same file name, with 'prompt.jsonl' replaced with 'samples.jsonl'
    """
    assert task_file_name.endswith("-prompt.jsonl")
    return task_file_name.replace("prompt.jsonl", "samples.jsonl")


def _generate_one_sample(prompt: str) -> str:
    """Sends a code completion prompt over to a language model and returns its
    response.

    Args:
        prompt (str): A prompt that contains at least one method signature and
        documentation describing its high-level specification, which an LLM is
        expected to provide a completion for.

    Returns:
        str: A language model's response to the given prompt (i.e., code
        completion task).
    """
    completion = llm_client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": CODE_GENERATION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )
    return completion.choices[0].message.content


args = parser.parse_args()
if not args.task:
    print(
        "Please supply a .jsonl file comprising a code completion task with the '--task' flag"
    )
    sys.exit(1)

# Note: The API exposed by the OpenAI test harness refers to prompts/tasks as
# 'problems'. We discontinue use of that word and instead refer to them as tasks.
tasks = read_problems(args.task)

samples = [
    dict(
        task_id=task_id,
        # The OpenAI evaluation harness expects a 'completion' rather than a
        # 'sample', which is why the result of `_generate_one_sample` is assigned
        # to a 'completion' key.
        completion=_generate_one_sample(tasks[task_id]["prompt"]),
    )
    for task_id in tasks
    for _ in range(NUM_SAMPLES_PER_TASK)
]
write_jsonl(_sample_file_name(args.task), samples)
