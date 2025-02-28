import os
import sys
import subprocess
import numpy as np
from typing import Any, Mapping
import json
import logging
import random

from utilities import musique_classes

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
        answer_records.append(
            musique_classes.Answer_Record(
                record.id, 
                record.answer if random.random() < 0.7 else "dummy", 
                [support.paragraph_support_idx for support in record.question_decomposition], 
                True
            )
        )

    with open(answer_path, 'w') as f:
        save_jsonl(f, answer_records)

    # evaluate result
    result = evaluate(answer_path, f"{musique_data_path}/musique_ans_v1.0_dev.jsonl")
    logging.info(result)