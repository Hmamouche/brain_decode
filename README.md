# Implementation code of the paper: A Multimodal LLM for the Non-Invasive Decoding of Spoken Text from Brain Recordings


## Requirements
```bash
python install -r requirement.txt
```

## Raw and Processed interaction data
* Download the data Raw and Processed from the following link and put in the main folder

## Tools
* Download vicuna-7b-v1.3 from https://huggingface.co/lmsys/vicuna-7b-v1.3 in the folder llm
* Download CLIP: git clone https://github.com/openai/CLIP

## Training Transformers
To train and test transformer models, from "benchmark_transformers" folder, run the same pyhton scripts but without '--test' the flag , for example:

```bash
python train.py --model-name [model name] --nb_seeds [number of runs (with different initialization seeds)]
```

```bash
train.py [-h] [--nb_seeds NB_SEEDS]
                [--model_name {Transformer,CNNTransformer,DuplexTransformerConv,BipartiteTransformerConv,DeconvBipartiteTransformerConv}]

Arguments:
  --nb_seeds NB_SEEDS, -ns NB_SEEDS
                        Number of seeds to execute.
  --model_name {Transformer,CNNTransformer,DuplexTransformerConv,BipartiteTransformerConv,DeconvBipartiteTransformerConv}
```   

## Testing Transformers
```bash
test.py [-h] [--nb_seeds NB_SEEDS]
               [--model_name {Transformer,CNNTransformer,DuplexTransformerConv,BipartiteTransformerConv,DeconvBipartiteTransformerConv}]

Arguments:
  --nb_seeds NB_SEEDS, -ns NB_SEEDS
                        Number of seeds. Must the same value used in training
  --model_name {Transformer,CNNTransformer,DuplexTransformerConv,BipartiteTransformerConv,DeconvBipartiteTransformerConv}

```  


## Training LLMs
To train and test Multimodal LLMs:

```bash
python trainer.py [-h] [--batch_size BATCH_SIZE]
                  [--model_name {MllmBrainToText,MllmBrainToTextV2}] [--test]
                  [--retrain] [--starting_epoch STARTING_EPOCH]
                  [--save_iters SAVE_ITERS] [--epochs EPOCHS]
                  [--saved_checkpoint SAVED_CHECKPOINT]

Arguments:
  --batch_size BATCH_SIZE
  --model_name {MllmBrainToText,MllmBrainToTextV2}, -m {MllmBrainToText,MllmBrainToTextV2}
                        Name of the model to train.
  --test                test the model
  --retrain             retrain from existing checkpoint
  --starting_epoch      starting epoch in case of retrain is True
  --save_epochs         number of epochs before saving the checkpoint
  --epochs              number of training epochs
  --saved_checkpoint    name of the checkpoint output file
```




## Evaluation / Benchmarking
```bash
python evaluation.py
```


## Additional Notes
* The structure of this repository is in work progress
* Some part of this code are inspired from InstructBlip (https://github.com/salesforce/LAVIS/blob/main/projects/instructblip/README.md), we thank the authors for their great work.
