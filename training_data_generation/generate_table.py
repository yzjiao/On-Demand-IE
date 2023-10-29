import args
import json
import time
import requests
import argparse
from tqdm import tqdm

API_KEY = ""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        default="dataset/paraphrased_instruction.json"
    )
    parser.add_argument(
        "--input_seed_file",
        type=str,
        default="dataset/seed.json"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=f"dataset/generated_table.json",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="",
        help="The API key to use. "
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
    prompt = "Given an information extraction instruction and the background text, extract the information as a markdown table. \n \
                Let this table include as many columns as possible. And keep the content brief.\n\n \
                The following are several examples:\n\n" 
    
    str_seed = ""
    for _id, seed in enumerate(seeds[:2]):
        str_seed += "Example %d:\n\nInstruction:\n%s\nText:\n%s\n\nTable:\n%s\n\n\n" % (_id + 1, seed['instruction'], seed['text'], seed['table'])
    
    content = prompt + str_seed
    content += "Following the format of the examples above, I would like you to help me extract the table for the following instruction and text: \n"
    content += "Instruction:\n" + instruction + "\nText:\n" + text
    messages = [{"role": "user", "content": content}]
    response_text = generate_chat_completion(messages)
    if response_text.startswith('Table: \n'):
        table = response_text[len('Table: \n'):]
    elif response_text.startswith('Table:\n'):
        table = response_text[len('Table:\n'):]
    elif response_text.startswith('Table:'):
        table = response_text[len('Table:'):]
    else:
        table = response_text
    return table, response_text


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
    API_KEY = args.api_key
    seed = load_data(args.input_seed_file)
    data = load_data(args.input_file)
    data_done = load_data(args.output_file)
    length, length_done = len(data), len(data_done)
    for i in tqdm(range(length_done, length)):
        dic = data[i]
        if 'instruction' in dic and 'text' in dic:
            instruction, text = dic['instruction'], dic['text']
            table, raw_table = generate_table(instruction, text, seed)
            dic['table'] = table
            dic['raw_table'] = raw_table
            data_done.append(dic)
            save_data(args.output_file, data_done)
    print('END.')


