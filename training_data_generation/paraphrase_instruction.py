import re
import args
import json
import time
import requests
import argparse
import copy
import random 
from tqdm import tqdm


import os
API_KEY = os.getenv("OPENAI_API_KEY")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file1",
        type=str,
        default="../dataset/generated_text.json"
    )
    parser.add_argument(
        "--input_file2",
        type=str,
        default=f"../dataset/generated_open_instruction.json"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=f"../dataset/paraphrased_instruction.json",
    )
    parser.add_argument(
        "--api_key",
        type=str,
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



def remove_numbered_sentences(sentences):
    result = []
    for sentence in sentences:
        stripped_sentence = sentence.strip()
        if re.match(r'^\d+\.', stripped_sentence):
            sentence_without_number = re.sub(r'^\d+\.', '', stripped_sentence)
            result.append(sentence_without_number.strip())
        else:
            result.append(stripped_sentence)
    return result

def remove_blank_lines(lines):
    non_blank_lines = [line for line in lines if line.strip()]
    return non_blank_lines

def paraphrase(instructions, chunk_size):
    style_list = ["real world user query", "professional request", "casual chat", "direct command"]
    style = random.choice(style_list) 
    prompt = "Given ten instructions, paraphrase them one by one in diffrent descriptive ways and make them like " + style + "real world user query but keep the key elements. So the outputs should ten paraphrased instructions. Remember not to output extra index or newline. \n\n"
    content = prompt + '\n'.join(instructions)
    messages = [{"role": "user", "content": content}]
    while True: 
        response_text = generate_chat_completion(messages)
        raw_lines = response_text.split('\n')
        raw_lines = remove_blank_lines(raw_lines)
        lines = remove_numbered_sentences(raw_lines)
        print(instructions)
        print(response_text)
        print(len(lines))
        print(chunk_size)
        if len(lines) == chunk_size:
            return lines, raw_lines


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

def chunk_list(input_list, chunk_size=10):
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]

def process_file(input_file, output_file, category):
    chunk_size = 10
    data = load_data(input_file)
    chunked_data = chunk_list(data, chunk_size)
    saved_data = load_data(args.output_file)
    length, length_saved = len(chunked_data), int(len(saved_data)/chunk_size)
    print("length_saved: ", length_saved * chunk_size)
    for i in tqdm(range(length_saved, length)):
        chunk = chunked_data[i]
        instructions = [dic['instruction'] for dic in chunk]
        outputs, raw_outputs = paraphrase(instructions, len(instructions))
        assert len(instructions) == len(outputs)
        assert len(instructions) == len(raw_outputs)
        for dic, output, raw_output in zip(chunk, outputs, raw_outputs):
            dic['instruction'] = output
            dic['raw_instruction'] = raw_output
            dic['category'] = category
            saved_data.append(dic)
            save_data(output_file, saved_data)



if __name__ == '__main__':
    args = parse_args()
    print(args.input_file1)
    process_file(args.input_file1, args.output_file, 'fixed_header')
    print(args.input_file2)
    process_file(args.input_file2, args.output_file, 'open_header')
    print('END.')


