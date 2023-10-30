# On-Demand-Information-Extraction
Official Repo for paper [Instruct and Extract: Instruction Tuning for On-Demand Information Extraction](https://arxiv.org/abs/2310.16040) by [Yizhu Jiao](https://yzjiao.github.io/), [Ming Zhong](https://maszhongming.github.io/), [Sha Li](https://raspberryice.github.io/), [Ruining Zhao](https://ruining0916.github.io/), [Ouyang Siru](https://ozyyshr.github.io/), [Heng Ji](http://blender.cs.illinois.edu/hengji.html) and [Jiawei Han](http://hanj.cs.illinois.edu/).


## :star2: Dataset

We release the InstructIE dataset including 14,579 samples for training ([dataset/training_data.json](https://github.com/yzjiao/On-Demand-IE/blob/main/dataset/training_data.json) and [dataset/training_data_cot.json](https://github.com/yzjiao/On-Demand-IE/blob/main/dataset/training_data_cot.json)) and 150 for testing ([dataset/test_data.json](https://github.com/yzjiao/On-Demand-IE/blob/main/dataset/test_data.json)). 
This instruction data can be used to conduct instruction-tuning for language models and make the language model follow instruction better in the task of on-demand information extraction. 


## :rocket: Data Generation from scratch

To generate the training data using your own seed tasks or other models, we open-source our scripts for the entire pipeline here. Our current code is tested on the GPT-3.5-turbo model accessible via the OpenAI API. 
**To run this pipeline**, Please find [the implementation details in the Data Generation directory](https://github.com/yzjiao/On-Demand-IE/tree/main/data_generation).



## :wrench: Model Training 
We finetune *LLaMA-7B* with *LoRA*, a parameter-efficient fine-tuning technique, on the training set of our *InstructIE* data to obtain the model *ODIE*. 
We format the datasets to follow a chatbot-style schema to allow interactions between the user and the language model into one input sequence.
During training, we compute the cross entropy loss.
Please find [more details about the training stage in the Training directory](https://github.com/yzjiao/On-Demand-IE/tree/main/training).


## :bar_chart: Model Evaluation
To evaluate the table header with semantics similarity, run the following script 
```bash
python evaluation/sim_for_header.py PATH_OF_FILE
```
To evaluate the table content with RougeL, run the following script 
```bash
python evaluation/rougel_for_content.py PATH_OF_FILE
```

Please provide the path of the evaluated file to run these two scripts. Otherwise, they would evaluate the output of ODIE, [model_output/ODIE-7b-filter.json](https://github.com/yzjiao/On-Demand-IE/blob/main/model_output/ODIE-7b-filter.json) by default. You can also try to evaluate the outputs of other models under [this directory](https://github.com/yzjiao/On-Demand-IE/tree/main/model_output). 


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