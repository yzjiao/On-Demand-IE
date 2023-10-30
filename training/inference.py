import os
import sys
import json
from tqdm import tqdm

import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

def load_json(path):
    with open(path) as f:
        data = json.loads(f.read())
    return data

def main(
    load_8bit: bool = False,
    base_model: str = "",
    use_chat_prompt: bool = True, # whether to use the prompt for multi-turn conversation
    lora_weights: str = "tloen/alpaca-lora-7b",
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        instruction,
        input=None,
        cot=False,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        # do_sample=True,
        max_new_tokens=2048,
        **kwargs,
    ):
        if use_chat_prompt == False:
            prompt = prompter.generate_prompt(instruction, input)
        else:
            # for cot
            if cot:
                prompt = "<|system|>\nYou are a helpful assistant. Follow the user instruction to output a paragraph as the explanation and extract information from the given text into a concise markdown table.\n\n"
            # for direct
            else:
                prompt = "<|system|>\nYou are a helpful assistant. Follow the user instruction to extract information from the given text into a concise markdown table.\n\n"
            if input == None:
                prompt += "<|user|>\n" + instruction.strip() + "\n\n<|assistant|>\n"
            else:
                prompt += "<|user|>\n" + instruction.strip() + "\n\n"+ input.strip() + "\n\n<|assistant|>\n"
        
        print(prompt)

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }
        
        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                # do_sample=do_sample,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s, skip_special_tokens=True)
        output = prompter.get_response(output, use_chat_prompt=use_chat_prompt)
        print(output)
        return output
    
    data_path = 'data/test.json'
    data = load_json(data_path)
    outputs = []

    for i in tqdm(range(len(data))):
        instruction = data[i]['instruction']
        text = data[i]['text']
        if 'open' in data[i]['category']:
            output = evaluate(instruction=instruction, input=text, cot=False)
        elif 'fixed' in data[i]['category']:
            output = evaluate(instruction=instruction, input=text, cot=False)
        else:
            raise ValueError('Wrong category!')

        cur = {}
        cur['instruction'] = instruction
        cur['text'] = text
        cur['gold'] = data[i]['table']
        cur['output'] = output
        cur['source_type'] = data[i]['source_type']
        cur['domain'] = data[i]['domain']
        cur['category'] = data[i]['category']
        cur['difficulty'] = data[i]['difficulty']
        
        outputs.append(cur)

    with open('output/{}.json'.format(lora_weights), 'w') as f:
        json.dump(outputs, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    fire.Fire(main)
