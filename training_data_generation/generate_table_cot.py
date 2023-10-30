import args
import json
import time
import requests
import argparse
from tqdm import tqdm

import os
API_KEY = os.getenv("OPENAI_API_KEY")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        default="../dataset/paraphrased_instruction.json"
    )
    parser.add_argument(
        "--input_seed_file",
        type=str,
        default="../dataset/seed_cot.json"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=f"../dataset/generated_table_cot.json",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="",
        help="The API key to use. If not specified, the key will be read from the environment variable OPENAI_API_KEY."
    )
    return parser.parse_args()


def generate_chat_completion(messages, model="gpt-3.5-turbo", temperature=0.6, max_tokens=None):
    API_ENDPOINT = "https://api.openai.com/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    if max_tokens is not None:
        data["max_tokens"] = max_tokens

    while True: 
        try:
            response = requests.post(API_ENDPOINT, headers=headers, data=json.dumps(data))
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                time.sleep(20)
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            time.sleep(20)


def generate_table(instruction, text, seeds):
    prompt = "Given an information extraction instruction and the background text, extract the information as a markdown table and produce a paragraph as the explanation. \
            Let this table include as many columns as possible and ensure the content is brief. \n\n \
            Please follow the below output format:\n \
            Explanation:\n \
            [a paragraph of what information should be extracted and why]\n \
            Table:\n \
            [a markdown table following the above instruction] \n\n \
            Here are several examples: \n\n "

    str_seed = ""
    for _id, seed in enumerate(seeds):
        str_seed += "Example %d:\n\nInstruction:\n%s\nText:\n%s\n\nExplanation:\n%s\nTable:\n%s\n\n" % (_id + 1, seed['instruction'], seed['text'], seed['explanation'], seed['table'])
    
    content = prompt + str_seed
    content += "Following the format of the examples above, I would like you to help me extract the table for the below instruction and text. Please adopt a step-by-step approach: generate a comprehensive explanation as the first step, followed by table extraction as the second step. \n"
    content += "Instruction: \n" + instruction + "\nText: \n" + text
    messages = [{"role": "user", "content": content}]
    while True:
        response_text = generate_chat_completion(messages)
        def split_by_substrings(main_string, substring1, substring2):
            parts = main_string.split(substring1)
            split_parts = [part.split(substring2) for part in parts]
            flattened_parts = [item for sublist in split_parts for item in sublist]
            flattened_parts = [i.strip('\n') for i in flattened_parts if i]
            return flattened_parts
        
        substr1, substr2 = 'Explanation:', 'Table:'
        parts = split_by_substrings(response_text, substr1, substr2)
        if len(parts) == 2:
            return parts, response_text


def load_data(fname):
    try:
        with open(fname, 'r') as file:
            saved_data = json.load(file)
    except FileNotFoundError:
        saved_data = []
    return saved_data


def save_data(output_file, data):
    with open(output_file, 'w') as file:
        file.write(json.dumps(data, indent=4))

if __name__ == '__main__':
    args = parse_args()
    seed = load_data(args.input_seed_file)
    data = load_data(args.input_file)
    data_done = load_data(args.output_file)
    length, length_done = len(data), len(data_done)
    print("length_done: ", length_done)
    for i in tqdm(range(length_done, length)):
        dic = data[i]
        if 'instruction' in dic and 'text' in dic:
            instruction, text = dic['instruction'], dic['text']
            [explanation, table], raw_output = generate_table(instruction, text, seed)
            dic['table'] = table
            dic['explanation'] = explanation
            dic['raw_output'] = raw_output
            data_done.append(dic)
            save_data(args.output_file, data_done)
    print('END.')


