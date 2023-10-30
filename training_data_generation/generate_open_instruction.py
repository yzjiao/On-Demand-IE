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
        "--input_seed_file",
        type=str,
        default="../dataset/seed.json"
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=2,
        help="if specified, only generate instance input for this many instructions",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=f"../dataset/generated_text.json"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=f"../dataset/generated_open_instruction.json",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="",
        help="The API key to use. If not specified, the key will be read from the environment variable OPENAI_API_KEY."
    )
    return parser.parse_args()


def generate_chat_completion(messages, model="gpt-3.5-turbo", temperature=1, max_tokens=None):
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


def gene_open_instr(seeds, text):
    prompt = "Give a background text, generate an instruction which mentions extracting the information from it. \
                But don't point out what kind of information should be extracted. \n \
                The following are several examples:\n\n" 
    
    str_seed = ""
    for _id, seed in enumerate(seeds):
        str_seed += "Example %d:\n\nText:\n%s\n\nInstruction:\n%s\n\n" % (_id + 1, seed['text'], seed['instruction'])

    content = prompt + str_seed
    content += "Following the format of the examples above, I would like you to help me generate the instruction for the following text:\nText:\n"
    content += text

    messages = [{"role": "user", "content": content}]
    response_text = generate_chat_completion(messages)

    if response_text.startswith('Instruction: \n'):
        text = response_text[len('Instruction: \n'):]
    elif response_text.startswith('Instruction:\n'):
        text = response_text[len('Instruction:\n'):]
    elif response_text.startswith('Instruction: '):
        text = response_text[len('Instruction: '):]
    else:
        text = response_text
    return text, response_text
    

def load_data(fname):
    try:
        with open(fname, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        data = []
    return data 

def save_data(output_file, data):
    with open(output_file, 'w') as file:
        file.write(json.dumps(data, indent=4))

if __name__ == '__main__':
    args = parse_args()
    seed = load_data(args.input_seed_file)[:args.num_seeds]
    filter_seed = []
    for data in seed:
        if 'category' in data and data['category'] == 'open header':
            filter_seed.append(data)

    data = load_data(args.input_file)
    data_done = load_data(args.output_file)
    length, length_done = len(data), len(data_done)
    print("length_done: ", length_done)
    for i in tqdm(range(length_done, length)):
        dic = data[i]
        if 'text' in dic:
            text = dic['text'] 
            instruction, raw_instruction = gene_open_instr(filter_seed, text)
            dic['instruction'] = instruction
            dic['raw_instruction'] = raw_instruction
            data_done.append(dic)
            save_data(args.output_file, data_done)

    print('END.')


