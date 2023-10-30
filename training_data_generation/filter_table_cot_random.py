import re
import sys
import args
import json
import time
import random
import requests
import argparse
from tqdm import tqdm
from tabulate import tabulate
from collections import defaultdict

sys.path.append('../UniEval')
from utils import convert_to_json
from metric.evaluator import get_evaluator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        default=f"../dataset/generated_table_cot.json"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=f"../dataset/filtered_table_cot.json",
    )
    parser.add_argument(
        "--model_input_file",
        type=str,
        default=f"../dataset/model_input_before_filter_cot.jsonl",
    )
    parser.add_argument(
        "--min_num_row",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--min_num_col",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--max_num_na",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--max_num_text_word",
        type=int,
        default=768,
    )
    parser.add_argument(
        "--min_consistency_sc",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--min_faithfulness_sc",
        type=float,
        default=0.5,
    )
    return parser.parse_args()

def load_data(fname):
    fin = open(fname, 'r')
    data = json.load(fin)
    return data

def save_data(output_file, data):
    with open(output_file, 'w') as file:
        file.write(json.dumps(data, indent=4))

def save_data_as_input(output_file, data):
    random.shuffle(data)
    with open(output_file, 'w') as file:
        for dic in data:
            instruction, domain, text, category, explanation = dic['instruction'], dic['domain'], dic['text'], dic['category'], dic['explanation']
            table = dic['table']
            sys_message = "You are a helpful assistant. Follow the user instruction to output a paragraph as the explanation and extract information from the given text into a concise markdown table." 
            user_message = instruction + '\n\n' + text
            ass_message = explanation + '\n\n' + table
            messages = [{"role": "system", "content": sys_message}, 
                        {"role": "user", "content": user_message}, 
                        {"role": "assistant", "content": ass_message}]
            
            new_dic = {}
            new_dic['instruction'], new_dic['domain'], new_dic['text'] = instruction, domain, text
            new_dic['category'], new_dic['table'], new_dic['messages'] = category, table, messages
            new_dic['explanation'] = explanation
            json_line = json.dumps(new_dic)
            file.write(json_line + '\n')

if __name__ == '__main__':
    args = parse_args()
    dataset = load_data(args.input_file)
    filterd_dataset = load_data(args.output_file)
    num_filter_data = len(filterd_dataset)
    print('num of raw data: ', len(filterd_dataset))
    unfilterd_dataset = random.choices(dataset, k=num_filter_data)
    for data in unfilterd_dataset:
        table = data['table']
        table = re.sub(' +', ' ', table)
        table = re.sub('\-{3,}', '---', table)
        data['table'] = table
    save_data_as_input(args.model_input_file, unfilterd_dataset)
    


