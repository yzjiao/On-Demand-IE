# Data Generation from scratch
To generate the training data using your own seed tasks or other models, we open-source our scripts for the entire pipeline here. Our current code is tested on the GPT-3.5-turbo model accessible via the OpenAI API.

**To run this pipeline**, first set your API Key:
```bash
# Obtain OpenAI API access from https://openai.com/blog/openai-api
# This key is necessary for all models since, by default, we use GPT-3.5-turbo for training data generation
export OPENAI_API_KEY='YOUR KEY HERE';
```

Additionally, you need to use an automatic evalautor, [UniEval](https://github.com/maszhongming/UniEval), for the step of data filtering in our pipeline. Please install the environment  for this tool. You can find more details in the [original repo](https://github.com/maszhongming/UniEval). 
```bash
git clone https://github.com/maszhongming/UniEval.git
cd UniEval
pip install -r requirements.txt
```


Now, you can run the following scripts for data generation:

```bash

cd data_generation

# 1. Generate fixed instructions from the seed tasks
python generate_fixed_instruction.py

# 2. Generate the text for each fixed instruction
python generate_text.py

# 3. Generate the open instruction for each text
python generate_open__instruction.py

# 4. Paraphrase all instructions 
python paraphrase_instruction.py

# 5. Generate table for each pair of instruction and text
python generate_table.py

# 6. Filtering, verifying, and reformatting
python filter_table.py

```
