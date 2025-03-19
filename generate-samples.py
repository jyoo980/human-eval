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

# Program arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--tasks",
    dest="tasks",
    type=str,
    help="The .jsonl file comprising code generation tasks",
)

client = OpenAI()


def generate_one_completion(prompt: str) -> str:
    completion = client.chat.completions.create(
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
        "Please supply a .jsonl file comprising code generation tasks with the '--tasks' flag"
    )
    sys.exit(1)

problems = read_problems(args.tasks)

num_samples_per_task = 10
samples = [
    dict(
        task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"])
    )
    for task_id in problems
    for _ in range(num_samples_per_task)
]
write_jsonl(f"{args.tasks}-samples.jsonl", samples)
