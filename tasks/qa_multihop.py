import os
import sys
import subprocess
import numpy as np
from typing import Any, Mapping
import json
import logging
import random
import argparse
import tempfile

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from quest_interface import Quest_Graph, Action
from implementations.language_agent import agent_functions, text_graph
from implementations.language_agent.persona import Persona

APP_ROOT = os.getenv("APP_ROOT", "/app")

with open(f"{APP_ROOT}/prompt_directory/react_tree.txt", "r") as file:
    prompt = file.read()

from utilities import install, musique_classes, language_models, embedding_models
from utilities.tokenizer import *

musique_path = f"{APP_ROOT}/cache/musique_data"
musique_repo_path = f"{musique_path}/repo"
musique_data_path = f"{musique_repo_path}/data"
os.makedirs(musique_path, exist_ok=True)
if len(os.listdir(musique_path)) == 0:
    subprocess.run(["git", "clone", "https://github.com/StonyBrookNLP/musique.git", musique_repo_path])
    # run bash download_data.sh, the result will be downloaded to musique_path/repo/data
    subprocess.run(["bash", "download_data.sh"], cwd=musique_repo_path)


def compute(record, verbose=False):

    working_memory = Quest_Graph(text_graph.Question_Node(record.question, None))
    persona = Persona(record.paragraphs, prompt)
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
    experiment_path = f"{APP_ROOT}/experiments/qa_multihop"
    answer_path = f"{experiment_path}/answers.jsonl"

    if args.reset:
        # remove the answer file
        if os.path.exists(answer_path):
            os.remove(answer_path)
        exit()

    os.makedirs(experiment_path, exist_ok=True)

    # load dataset,
    # use musique_ans_v1.0_train.jsonl to train
    # use musique_ans_v1.0_dev.jsonl to validate results
    # use musique_ans_v1.0_test.jsonl to generate submission results
    train_task = musique_classes.Task(f"{musique_data_path}/musique_ans_v1.0_train.jsonl", answer_path)
    validate_task = musique_classes.Task(f"{musique_data_path}/musique_ans_v1.0_dev.jsonl", answer_path)
    test_task = musique_classes.Task(f"{musique_data_path}/musique_ans_v1.0_test.jsonl", answer_path)

    i = 0
    while validate_task.has_next():
        record_id, record = validate_task.pop()
        answer_record = compute(record, verbose=False)
        validate_task.fulfill(record_id, answer_record)
        i += 1
        if i % 100 == 0:
            logging.info(validate_task.status())

    # evaluate result
    # write answer to tempfile
    with tempfile.NamedTemporaryFile(mode="w", delete=True, delete_on_close=True) as temp_file:
        validate_task.write_answer(temp_file)
        eval_path = temp_file.name
        # python evaluate_v1.0.py <eval_path> <dev_file_path>
        console_result = subprocess.run(["python", f"{musique_repo_path}/evaluate_v1.0.py", eval_path, validate_task.file_path], capture_output=True)
        if console_result.returncode != 0:
            logging.error(f"Error in evaluation: {console_result.stderr.decode()}")
        else:
            output_str = console_result.stdout.decode()
            json_obj = json.loads(output_str)
            f1 = json_obj["answer_f1"]
            em = json_obj["answer_em"]
            support_f1 = json_obj["support_f1"]
            logging.info(f"F1: {f1}, EM: {em}, Support F1: {support_f1}")