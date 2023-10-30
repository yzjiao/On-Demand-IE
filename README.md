# On-Demand-Information-Extraction
Official Repo for paper [Instruct and Extract: Instruction Tuning for On-Demand Information Extraction](https://arxiv.org/abs/2310.16040) by [Yizhu Jiao](https://yzjiao.github.io/), [Ming Zhong](https://maszhongming.github.io/), [Sha Li](https://raspberryice.github.io/), [Ruining Zhao](https://ruining0916.github.io/), [Ouyang Siru](https://ozyyshr.github.io/), [Heng Ji](http://blender.cs.illinois.edu/hengji.html) and [Jiawei Han](http://hanj.cs.illinois.edu/).


## :star2: Dataset

We release the InstructIE dataset including 14,579 samples for training ([dataset/training_data.json](https://github.com/yzjiao/On-Demand-IE/blob/main/dataset/training_data.json) and [dataset/training_data_cot.json](https://github.com/yzjiao/On-Demand-IE/blob/main/dataset/training_data_cot.json)) and 150 for testing ([dataset/test_data.json](https://github.com/yzjiao/On-Demand-IE/blob/main/dataset/test_data.json)). 
This instruction data can be used to conduct instruction-tuning for language models and make the language model follow instruction better in the task of on-demand information extraction. 


## :rocket: Data Generation from scratch

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


## :wrench: Model Training 
We finetune LLaMA-7B with LoRA, a parameter-efficient fine-tuning technique, on the training set of our InstructIE data to obtain the model \textsc{Odie}. 
We format the datasets to follow a chatbot-style schema to allow interactions between the user and the language model into one input sequence.
During training, we compute the cross entropy loss.
Please find more details about the training stage in [https://github.com/yzjiao/On-Demand-IE/tree/main/training]{https://github.com/yzjiao/On-Demand-IE/tree/main/training}.



## :books: Citation

If you find this repo helpful, please cite our paper:

```bibtex
@article{jiao2023instruct,
  title={Instruct and Extract: Instruction Tuning for On-Demand Information Extraction},
  author={Jiao, Yizhu and Zhong, Ming and Li, Sha and Zhao, Ruining and Ouyang, Siru and Ji, Heng and Han, Jiawei},
  journal={arXiv preprint arXiv:2310.16040},
  year={2023}
}
```