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
        default=f"../dataset/dataset.json"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=f"../dataset/processed_dataset.json",
    )
    parser.add_argument(
        "--model_input_file",
        type=str,
        default=f"../dataset/model_input_before_filter.jsonl",
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
        file.write(json.dumps(data, indent=4, ensure_ascii=True))

def is_markdown_table(input_str):
    def remove_before_character(text, character):
        index = text.find(character)
        if index != -1:
            return text[index:]
        return ""

    def remove_after_last_character(text, character):
        index = text.rfind(character)
        if index != -1:
            return text[:index+1]
        return text

    input_str = remove_before_character(input_str, '|')
    input_str = remove_after_last_character(input_str, '|')
    
    lines = []
    raw_lines = input_str.strip().split('\n')
    for line in raw_lines:
        line = remove_before_character(line, '|')
        line = remove_after_last_character(line, '|')
        lines.append(line)

    if len(lines) < 3:
        return False, ''

    header = lines[0].strip()
    if not (header.startswith('|') and header.endswith('|')):
        return False, ''

    column_count = lines[0].count('|')
    column_count
    for line in lines[2:]:
        if not (line.startswith('|') and line.endswith('|')) or line.count('|') != column_count:
            return False, ''
    return True, '\n'.join(lines)


def save_data_as_input(output_file, data):
    with open(output_file, 'w') as file:
        for dic in data:
            instruction, domain, text, category, table = dic['instruction'], dic['domain'], dic['text'], dic['category'], dic['table']

            sys_message = "You are a helpful assistant. Follow the user instruction to extract information from the given text into a concise markdown table." 
            user_message = instruction + '\n\n' + text
            ass_message = table
            messages = [{"role": "system", "content": sys_message}, 
                        {"role": "user", "content": user_message}, 
                        {"role": "assistant", "content": ass_message}]
            
            new_dic = {}
            new_dic['instruction'], new_dic['domain'], new_dic['text'] = instruction, domain, text
            new_dic['category'], new_dic['table'], new_dic['messages'] = category, table, messages

            json_line = json.dumps(new_dic)
            file.write(json_line + '\n')

if __name__ == '__main__':
    args = parse_args()
    dataset = load_data(args.input_file)
    num_data = len(dataset)
    print('num of data: ', len(dataset))
    for data in dataset:
        table = data['table']
        flag, table_str = is_markdown_table(table)
        if flag is False:
            print(data['instruction'])

        table = table.replace('|', ' | ')
        table = table.replace('| \n |', '|\n|') 
        table = re.sub(' +', ' ', table)
        table = re.sub('\-{3,}', '---', table)
        table = table.replace('| |', '| N/A |')
        table = table.replace('| Not specified |', '| N/A |')
        table = table.replace('| Not Specified |', '| N/A |')
        data['table'] = table.strip()
        data['text'] = data['text'].strip()
        data['instruction'] = data['instruction'].strip()
        
        if 'gpt4_output' in data:
            del data['gpt4_output']
        
    save_data(args.output_file, dataset)


    
    


