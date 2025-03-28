from human_eval.data import write_jsonl, read_problems
from openai import OpenAI

import argparse
import sys

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
    "--tasks",
    dest="tasks",
    type=str,
    help="The .jsonl file comprising code generation tasks",
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


def _generate_one_completion(prompt: str) -> str:
    """Sends a code completion prompt over to a language model and returns its
    response.

    Args:
        prompt (str): A prompt containing a code completion task.

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
if not args.tasks:
    print(
        "Please supply a .jsonl file comprising code completion tasks with the '--tasks' flag"
    )
    sys.exit(1)

problems = read_problems(args.tasks)

samples = [
    dict(
        task_id=task_id,
        completion=_generate_one_completion(problems[task_id]["prompt"]),
    )
    for task_id in problems
    for _ in range(NUM_SAMPLES_PER_TASK)
]
write_jsonl(_sample_file_name(args.tasks), samples)
