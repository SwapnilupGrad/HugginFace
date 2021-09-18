---
language:
- en
license: apache-2.0
tags:
- generated_from_trainer
datasets:
- glue
metrics:
- accuracy
- f1
model-index:
- name: bert-base-cased-finetuned-qqp
  results:
  - task:
      name: Text Classification
      type: text-classification
    dataset:
      name: GLUE QQP
      type: glue
      args: qqp
    metrics:
    - name: Accuracy
      type: accuracy
      value: 0.9083848627256987
    - name: F1
      type: f1
      value: 0.8767633750332712
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# bert-base-cased-finetuned-qqp

This model is a fine-tuned version of [bert-base-cased](https://huggingface.co/bert-base-cased) on the GLUE QQP dataset.
It achieves the following results on the evaluation set:
- Loss: 0.3752
- Accuracy: 0.9084
- F1: 0.8768
- Combined Score: 0.8926

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 16
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3.0

### Training results

| Training Loss | Epoch | Step  | Validation Loss | Accuracy | F1     | Combined Score |
|:-------------:|:-----:|:-----:|:---------------:|:--------:|:------:|:--------------:|
| 0.308         | 1.0   | 22741 | 0.2548          | 0.8925   | 0.8556 | 0.8740         |
| 0.201         | 2.0   | 45482 | 0.2881          | 0.9032   | 0.8698 | 0.8865         |
| 0.1416        | 3.0   | 68223 | 0.3752          | 0.9084   | 0.8768 | 0.8926         |


### Framework versions

- Transformers 4.11.0.dev0
- Pytorch 1.9.0
- Datasets 1.12.1
- Tokenizers 0.10.3
