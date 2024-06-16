# Implementation code of the paper: A Multimodel LLM for the Non-Invasive Decoding of Spoken Text from Brain Recordings


## Requirements
* fasttext==0.9.2
* nltk==3.7
* numpy==1.26.4
* pandas==1.4.2
* torch==2.1.2
* transformers==4.33.2
* auto_mix_prep==0.2.0
* wandb==0.16.3

```bash
python install -r requirement.txt
```


## Training Transformers
To train and test transformer models, from "benchmark_transformers" folder, run the same pyhton scripts but without '--test' the flag , for example:

```bash
python train.py --model-name [model name] --nb_seeds [number of runs (with different initialization seeds)]
```

```bash
train.py [-h] [--nb_seeds NB_SEEDS]
                [--model_name {Transformer,CNNTransformer,DuplexTransformerConv,BipartiteTransformerConv,DeconvBipartiteTransformerConv}]

Arguments:
  -h, --help            show this help message and exit
  --nb_seeds NB_SEEDS, -ns NB_SEEDS
                        Number of seeds to execute.
  --model_name {Transformer,CNNTransformer,DuplexTransformerConv,BipartiteTransformerConv,DeconvBipartiteTransformerConv}
```   

## Testing Transformers
```bash
test.py [-h] [--nb_seeds NB_SEEDS]
               [--model_name {Transformer,CNNTransformer,DuplexTransformerConv,BipartiteTransformerConv,DeconvBipartiteTransformerConv}]

Arguments:
  -h, --help            show this help message and exit
  --nb_seeds NB_SEEDS, -ns NB_SEEDS
                        Number of seeds. Must the same value used in training
  --model_name {Transformer,CNNTransformer,DuplexTransformerConv,BipartiteTransformerConv,DeconvBipartiteTransformerConv}

```  


## Training LLMs
To train and test Multimodal LLMs:

```bash
python  mllm_frmi_to_txt_vicuna.py
```

And for the version that includes image in the input:

```bash
python  mllm_image_frmi_to_txt_vicuna.py
```




## Evaluation / Benchmarking
```bash
python evaluation.py
```


## Additional Notes
* The structure of this repository is in work progress
