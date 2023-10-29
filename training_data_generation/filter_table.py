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
        default=f"dataset/generated_table.json"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=f"dataset/filtered_table.json",
    )
    parser.add_argument(
        "--model_input_file",
        type=str,
        default=f"dataset/model_input_after_filter.jsonl",
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

def save_data_as_model_input(output_file, data):
    random.shuffle(data)
    with open(output_file, 'w') as file:
        for dic in data:
            instruction, domain, text, category = dic['instruction'], dic['domain'], dic['text'], dic['category']
            table = list_of_dicts_to_markdown_table(dic['table_dic'])

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

    separator_line = lines[1].strip()
    if not re.match(r'^\|[-|\s]+\|$', separator_line):
        return False, ''

    column_count = lines[0].count('|')
    column_count
    for line in lines[2:]:
        if not (line.startswith('|') and line.endswith('|')) or line.count('|') != column_count:
            return False, ''
    return True, '\n'.join(lines)


def markdown2dic(input_str):
    rows = input_str.split("\n")
    rows = [row.strip() for row in rows if row]
    keys = [key.strip() for key in rows[0].split("|") if key]
    data = []
    for row in rows[2:]: 
        values = [value.strip() for value in row.split("|") if value]
        data.append(dict(zip(keys, values)))
    return data, len(rows)-2, len(keys)


def if_consistent_with_instruction(instructions, headers):
    src_list = instructions
    output_list = []
    for header in headers:
        str = ['Extract the ' + col + ' from the given text.' for col in header]
        output_list.append(' '.join(str))
    data = convert_to_json(output_list=output_list, src_list=src_list)
    evaluator = get_evaluator('fact')
    eval_scores = evaluator.evaluate(data, print_result=False)
    return [dic['consistency'] for dic in eval_scores]


def if_faithful_to_text(texts, tables):
    src_list = texts
    output_list = []
    for table in tables:
        sentences = []
        for dic in table:
            keys = list(dic.keys())
            row_name = keys[0]
            for key in keys[1:]:
                value = dic[key]
                sentence = f'The {key} of {dic[row_name]} is {value}.'
                sentences.append(sentence)
        output_list.append(' '.join(sentences))
    data = convert_to_json(output_list=output_list, src_list=src_list)
    evaluator = get_evaluator('fact')
    eval_scores = evaluator.evaluate(data, print_result=False)
    return [dic['consistency'] for dic in eval_scores]


def list_of_dicts_to_markdown_table(data):
    headers = data[0].keys()
    header_line = " | ".join(list(headers))
    separator_line = " | ".join(['---' for header in headers])
    data_lines = []
    for item in data:
        row = " | ".join(list(item.values()))
        data_lines.append(row)

    table = f"| {header_line} |\n| {separator_line} |\n" + "\n".join(f"| {line} |" for line in data_lines)
    return table



if __name__ == '__main__':
    args = parse_args()
    data = load_data(args.input_file)
    print('num of raw data: ', len(data))
    valid_data = []
    for dic in tqdm(data):
        flag = True
        if 'instruction' in dic and 'text' in dic and 'table' in dic:
            instruction, text, table_str, category = dic['instruction'], dic['text'], dic['table'], dic['category']
            flag, processed_str = is_markdown_table(table_str)
            if flag: 
                dic['table'] = processed_str
                valid_data.append(dic)
    print('num of valid data: ', len(valid_data))


    informative_data = []
    for dic in tqdm(valid_data):
        instruction, text, table_str, category = dic['instruction'], dic['text'], dic['table'], dic['category']
        table_dic, num_row, num_col = markdown2dic(table_str)
        dic['table_dic'] = table_dic
        if num_row + num_col > 3 and num_col > 1:
        # if num_row >= args.min_num_row and num_col >= args.min_num_col:
            if table_str.count('N/A') < args.max_num_na: 
                if len(text.split(' ')) <= args.max_num_text_word:
                    informative_data.append(dic)
    print('num of informative data: ', len(informative_data))
    
    fixed_header_data = [dic for dic in informative_data if dic['category'] == "fixed_header"]
    open_header_data = [dic for dic in informative_data if dic['category'] == "open_header"]
    instructions = [dic['instruction'] for dic in fixed_header_data]
    headers = [dic['table_dic'][0].keys() for dic in fixed_header_data]
    consistency_sc = if_consistent_with_instruction(instructions, headers)
    # print(consistency_sc)
    
    consistent_data = []
    threshold = args.min_consistency_sc
    for dic, sc in zip(fixed_header_data, consistency_sc):
        if sc >= threshold:
            consistent_data.append(dic)
    consistent_data += open_header_data
    print('num of consistent data: ', len(consistent_data))

    consistent_data = consistent_data
    threshold = args.min_faithfulness_sc
    texts = [dic['text'] for dic in consistent_data]
    table_dics = [dic['table_dic'] for dic in consistent_data]
    faithful_sc = if_faithful_to_text(texts, table_dics)
    faithful_data = []
    for dic, sc in zip(consistent_data, faithful_sc):
        if sc >= threshold:
            faithful_data.append(dic)
    print('num of faithful data: ', len(faithful_data))

    save_data(args.output_file, faithful_data)
    save_data_as_model_input(args.model_input_file, faithful_data)
    
    print('END.')
    


