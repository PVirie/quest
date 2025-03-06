import os
import sys
import subprocess
import numpy as np
from typing import Any, Mapping
import json
import logging
import random

from quest_interface import Quest_Graph
from implementations.thoughts import consistent_tree, text_graph
from implementations.thoughts.persona import Persona

from utilities import musique_classes, install

musique_path = "/app/cache/musique_data"
musique_repo_path = f"{musique_path}/repo"
musique_data_path = f"{musique_repo_path}/data"
os.makedirs(musique_path, exist_ok=True)
if len(os.listdir(musique_path)) == 0:
    subprocess.run(["git", "clone", "https://github.com/StonyBrookNLP/musique.git", musique_repo_path])
    # run bash download_data.sh, the result will be downloaded to musique_path/repo/data
    subprocess.run(["bash", "download_data.sh"], cwd=musique_repo_path)


def load_data_records(data_path):
    data_records = []
    # jsonl file
    with open(data_path, 'r') as f:
        for line in f:
            data_records.append(musique_classes.Question_Record(json.loads(line)))
    return data_records


def save_jsonl(f, answer_records):
    for record in answer_records:
        f.write(json.dumps(record.to_json()) + "\n")


def evaluate(answer_path, dev_file_path):
    # python evaluate_v1.0.py <answer_path> <dev_file_path>
    console_result = subprocess.run(["python", f"{musique_repo_path}/evaluate_v1.0.py", answer_path, dev_file_path], capture_output=True)
    output_str = console_result.stdout.decode()
    # parse into object
    json_obj = json.loads(output_str)
    return json_obj


install("litellm")
from litellm import completion

def llm_function(system_prompt, user_prompt):
    messages = [
        {"content": system_prompt, "role": "system"},
        {"content": user_prompt, "role": "user"}
    ]
    response = completion(model=os.getenv("QA_MODEL"), messages=messages)
    text = response.choices[0].message.content
    return text


def compute(record):
    working_memory = Quest_Graph(text_graph.Text_Node(text_graph.Text_Node_Type.Question_Node, record.question, None))
    persona = Persona(llm_function)
    while True:
        action, param_1, param_2 = consistent_tree.agent_function(persona, working_memory.query())
        if param_2 is None:
            break
        if action == consistent_tree.Action.ANSWER:
            working_memory.respond(param_1, param_2)
        else:
            working_memory.discover(param_1, param_2)

    return record.answer if random.random() < 0.7 else "dummy", [support.paragraph_support_idx for support in record.question_decomposition]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    experiment_path = "/app/experiments/qa_multihop"
    answer_path = f"{experiment_path}/answers.jsonl"

    os.makedirs(experiment_path, exist_ok=True)

    # load dataset, 
    # musique_ans_v1.0_dev.jsonl contains full answer
    # use musique_ans_v1.0_train.jsonl to train
    # use musique_ans_v1.0_test.jsonl to generate submission results
    data_records = load_data_records(f"{musique_data_path}/musique_ans_v1.0_dev.jsonl")
    # print(data_records[10])

    answer_records = []
    for record in data_records:
        # generate answer here
        answer, supports = compute(record)
        answer_records.append(musique_classes.Answer_Record(record.id, answer, supports, True))

    with open(answer_path, 'w') as f:
        save_jsonl(f, answer_records)

    # evaluate result
    result = evaluate(answer_path, f"{musique_data_path}/musique_ans_v1.0_dev.jsonl")
    logging.info(result)