import args
import json
import time
import requests
import argparse

API_KEY = ""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        default="dataset/seed.json"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="dataset/generated_instruction.json",
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=5,
        help="if specified, only generate instance input for this many instructions",
    )
    parser.add_argument(
        "--max_instruction",
        type=int,
        default=10000,
        help="The max number of instructions to generate.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="",
        help="The API key to use. "
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
        response = requests.post(API_ENDPOINT, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            time.sleep(20)
            # raise Exception(f"Error {response.status_code}: {response.text}")

def generate_instruction(seeds):
    prompt = "Now I want to generate some real-world examples of information extraction that users would let the AI help with.\n  \
                Specifically, an example should contain the following two items: \n \
                1. Instruction: User input, usually refers to extracting some desired information from a given text. \n \
                2. Domain: The domain to which the user query belongs. \n\n \
                The following are several examples:\n\n" 
    
    str_seed = ""
    for _id, seed in enumerate(seeds):
        str_seed += "Example %d:\nInstruction: \"%s\"\nDomain: \"%s\"\n\n" % (_id + 1, seed['instruction'], seed['domain'])
    #print(str_seed)

    content = prompt + str_seed
    content += "Following the format of the examples above, I would like you to help me generate ten more new examples that meet the following requirements: \n \
                1. These examples should be in various domains. \n \
                2. These examples should be described in different styles. \n \
                3. The generated domains do not overlap with the above example. \n\n "

    messages = [{"role": "user", "content": content}]
    response_text = generate_chat_completion(messages)

    new_data = []
    idx = 0
    lines = response_text.split('\n')
    while idx + 2 < len(lines):
        if lines[idx].startswith('Example') and lines[idx + 1].startswith('Instruction: "') and lines[idx + 2].startswith('Domain: "'):
            dic = {"instruction": lines[idx + 1][14:-1], "domain": lines[idx + 2][9:-1]}
            new_data.append(dic)
            idx += 4
        else:
            idx += 1
    return new_data


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
    seeds = load_data(args.input_file)[:args.num_seeds]
    all_data  = load_data(args.output_file)
    print(args.max_instruction)
    while len(all_data) < args.max_instruction:
        new_data = generate_instruction(seeds)
        all_data += new_data
        save_data(args.output_file, all_data)
        print(len(all_data))
    print('END.')


