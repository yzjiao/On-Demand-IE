import os
import json
import random
from os.path import join

merge_list = ['cot', 'gpt4_alpaca', 'code_alpaca']

def load_jsonl(path):
    cur_data = []
    with open(path) as f:
        for line in f:
            cur_data.append(json.loads(line))
    return cur_data

data = []
for dataset in merge_list:
    path = join('../data', dataset)
    files = os.listdir(path)
    for cur_file in files:
        if '.jsonl' not in cur_file:
            continue
        data += load_jsonl(join(path, cur_file))

# v1: cot + gpt4_alpaca + code_alpaca
random.shuffle(data)
with open('../data/mixed_data_v1.jsonl', 'w') as f:
    for i in range(len(data)):
        print(json.dumps(data[i]), file=f)
        
        
