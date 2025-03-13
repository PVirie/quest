import os
import sys
import subprocess
import numpy as np
from typing import Any, Mapping
import json
import logging
import random
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from quest_interface import Quest_Graph, Action
from implementations.thoughts import agent_functions, text_graph
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



def compute(record, verbose=False):

    working_memory = Quest_Graph(text_graph.Question_Node(record.question, None))
    persona = Persona(record.paragraphs)
    while True:
        action, param_1, param_2 = agent_functions.basic_tree(persona, working_memory.query())
        if action == Action.ANSWER:
            working_memory.respond(param_1, param_2)
            if param_2 is None:
                if param_1 is None:
                    break
                # answering the root
                # retrieve answer and support paragraph indices from the working memory
                answer = working_memory.root.answer
                support_paragraph_indices = working_memory.root.gather_support_ids()
                return musique_classes.Answer_Record(record.id, answer.strip(), list(support_paragraph_indices), True)
        elif action == Action.DISCOVER:
            working_memory.discover(param_1, param_2)
            if len(working_memory) > 20:
                break
        else:
            raise ValueError("Invalid action")
        
        if verbose:
            pass
        
    return musique_classes.Answer_Record(record.id, "Cannot find answer.", [], False)


if __name__ == "__main__":
    # optional flag --reset or -r, default is false
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", "-r", action="store_true")
    args = parser.parse_args()

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

    start = 0
    if not args.reset and os.path.exists(answer_path):
        # continue from the last record
        with open(answer_path, 'r') as f:
            for line in f:
                start += 1

    for i, record in enumerate(data_records[start:], start):
        # generate answer here
        answer_record = compute(record, verbose=False)
        with open(answer_path, 'a') as f:
            f.write(json.dumps(answer_record.to_json()) + "\n")

        # print every 100 records
        if i % 100 == 0:
            logging.info(f"Record {i}/{len(data_records)} done")

    # evaluate result
    result = evaluate(answer_path, f"{musique_data_path}/musique_ans_v1.0_dev.jsonl")
    logging.info(result)