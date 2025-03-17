from human_eval.data import write_jsonl, read_problems
from openai import OpenAI

CODE_GENERATION_SYSTEM_PROMPT = """
Given a Python method with a docstring, generate only the completion for the
method. That is, do not repeat any of the method signature or docstring;
generate only the code that should go in the method body.
"""

DEFAULT_MODEL="gpt-4o-mini-2024-07-18",

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
    model_result = completion.choices[0].message.content
    print("Result")
    print(model_result)
    return model_result


problems = read_problems("data/humaneval-tagged-format-mini.jsonl.gz")

num_samples_per_task = 50
samples = [
    dict(
        task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"])
    )
    for task_id in problems
    for _ in range(num_samples_per_task)
]
write_jsonl("foo.jsonl", samples)
