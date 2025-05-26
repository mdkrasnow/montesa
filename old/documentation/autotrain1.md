AutoTrain documentation

Quickstart with Python

# AutoTrain

🏡 View all docsAWS Trainium & InferentiaAccelerateAmazon SageMakerArgillaAutoTrainBitsandbytesChat UIDataset viewerDatasetsDiffusersDistilabelEvaluateGradioHubHub Python LibraryHuggingface.jsInference Endpoints (dedicated)Inference ProvidersLeaderboardsLightevalOptimumPEFTSafetensorsSentence TransformersTRLTasksText Embeddings InferenceText Generation InferenceTokenizersTransformersTransformers.jssmolagentstimm

Search documentation
`Ctrl+K`

mainv0.8.24v0.7.129v0.6.48v0.5.2EN

[4,366](https://github.com/huggingface/autotrain-advanced)

You are viewing main version, which requires installation from source. If you'd like
regular pip install, checkout the latest stable version ( [v0.8.24](https://huggingface.co/docs/autotrain/v0.8.24/quickstart_py)).


![Hugging Face's logo](https://huggingface.co/front/assets/huggingface_logo-noborder.svg)

Join the Hugging Face community

and get access to the augmented documentation experience


Collaborate on models, datasets and Spaces


Faster examples with accelerated inference


Switch between documentation themes


[Sign Up](https://huggingface.co/join)

to get started

# Quickstart with Python

AutoTrain is a library that allows you to train state of the art models on Hugging Face Spaces, or locally.
It provides a simple and easy-to-use interface to train models for various tasks like llm finetuning, text classification,
image classification, object detection, and more.

In this quickstart guide, we will show you how to train a model using AutoTrain in Python.

## Getting Started

AutoTrain can be installed using pip:

Copied

```
$ pip install autotrain-advanced
```

The example code below shows how to finetune an LLM model using AutoTrain in Python:

Copied

```
import os

from autotrain.params import LLMTrainingParams
from autotrain.project import AutoTrainProject

params = LLMTrainingParams(
    model="meta-llama/Llama-3.2-1B-Instruct",
    data_path="HuggingFaceH4/no_robots",
    chat_template="tokenizer",
    text_column="messages",
    train_split="train",
    trainer="sft",
    epochs=3,
    batch_size=1,
    lr=1e-5,
    peft=True,
    quantization="int4",
    target_modules="all-linear",
    padding="right",
    optimizer="paged_adamw_8bit",
    scheduler="cosine",
    gradient_accumulation=8,
    mixed_precision="bf16",
    merge_adapter=True,
    project_name="autotrain-llama32-1b-finetune",
    log="tensorboard",
    push_to_hub=True,
    username=os.environ.get("HF_USERNAME"),
    token=os.environ.get("HF_TOKEN"),
)

backend = "local"
project = AutoTrainProject(params=params, backend=backend, process=True)
project.create()
```

In this example, we are finetuning the `meta-llama/Llama-3.2-1B-Instruct` model on the `HuggingFaceH4/no_robots` dataset.
We are training the model for 3 epochs with a batch size of 1 and a learning rate of `1e-5`.
We are using the `paged_adamw_8bit` optimizer and the `cosine` scheduler.
We are also using mixed precision training with a gradient accumulation of 8.
The final model will be pushed to the Hugging Face Hub after training.

To train the model, run the following command:

Copied

```
$ export HF_USERNAME=<your-hf-username>
$ export HF_TOKEN=<your-hf-write-token>
$ python train.py
```

This will create a new project directory with the name `autotrain-llama32-1b-finetune` and start the training process.
Once the training is complete, the model will be pushed to the Hugging Face Hub.

Your HF\_TOKEN and HF\_USERNAME are only required if you want to push the model or if you are accessing a gated model or dataset.

## AutoTrainProject Class

### classautotrain.project.AutoTrainProject

[<source>](https://github.com/huggingface/autotrain-advanced/blob/main/src/autotrain/project.py#L443)

(params: typing.Union\[autotrain.trainers.clm.params.LLMTrainingParams, autotrain.trainers.text\_classification.params.TextClassificationParams, autotrain.trainers.tabular.params.TabularParams, autotrain.trainers.seq2seq.params.Seq2SeqParams, autotrain.trainers.image\_classification.params.ImageClassificationParams, autotrain.trainers.text\_regression.params.TextRegressionParams, autotrain.trainers.object\_detection.params.ObjectDetectionParams, autotrain.trainers.token\_classification.params.TokenClassificationParams, autotrain.trainers.sent\_transformers.params.SentenceTransformersParams, autotrain.trainers.image\_regression.params.ImageRegressionParams, autotrain.trainers.extractive\_question\_answering.params.ExtractiveQuestionAnsweringParams, autotrain.trainers.vlm.params.VLMTrainingParams\]backend: strprocess: bool = False)

A class to train an AutoTrain project

## Attributes

params : Union\[\
LLMTrainingParams,\
TextClassificationParams,\
TabularParams,\
Seq2SeqParams,\
ImageClassificationParams,\
TextRegressionParams,\
ObjectDetectionParams,\
TokenClassificationParams,\
SentenceTransformersParams,\
ImageRegressionParams,\
ExtractiveQuestionAnsweringParams,\
VLMTrainingParams,\
\]
The parameters for the AutoTrain project.
backend : str
The backend to be used for the AutoTrain project. It should be one of the following:

- local
- spaces-a10g-large
- spaces-a10g-small
- spaces-a100-large
- spaces-t4-medium
- spaces-t4-small
- spaces-cpu-upgrade
- spaces-cpu-basic
- spaces-l4x1
- spaces-l4x4
- spaces-l40sx1
- spaces-l40sx4
- spaces-l40sx8
- spaces-a10g-largex2
- spaces-a10g-largex4
process : bool
Flag to indicate if the params and dataset should be processed. If your data format is not AutoTrain-readable, set it to True. Set it to True when in doubt. Defaults to False.

## Methods

**post\_init**():
Validates the backend attribute.
create():
Creates a runner based on the backend and initializes the AutoTrain project.

## Parameters

### Text Tasks

### classautotrain.trainers.clm.params.LLMTrainingParams

[<source>](https://github.com/huggingface/autotrain-advanced/blob/main/src/autotrain/trainers/clm/params.py#L8)

(model: str = 'gpt2'project\_name: str = 'project-name'data\_path: str = 'data'train\_split: str = 'train'valid\_split: typing.Optional\[str\] = Noneadd\_eos\_token: bool = Trueblock\_size: typing.Union\[int, typing.List\[int\]\] = -1model\_max\_length: int = 2048padding: typing.Optional\[str\] = 'right'trainer: str = 'default'use\_flash\_attention\_2: bool = Falselog: str = 'none'disable\_gradient\_checkpointing: bool = Falselogging\_steps: int = -1eval\_strategy: str = 'epoch'save\_total\_limit: int = 1auto\_find\_batch\_size: bool = Falsemixed\_precision: typing.Optional\[str\] = Nonelr: float = 3e-05epochs: int = 1batch\_size: int = 2warmup\_ratio: float = 0.1gradient\_accumulation: int = 4optimizer: str = 'adamw\_torch'scheduler: str = 'linear'weight\_decay: float = 0.0max\_grad\_norm: float = 1.0seed: int = 42chat\_template: typing.Optional\[str\] = Nonequantization: typing.Optional\[str\] = 'int4'target\_modules: typing.Optional\[str\] = 'all-linear'merge\_adapter: bool = Falsepeft: bool = Falselora\_r: int = 16lora\_alpha: int = 32lora\_dropout: float = 0.05model\_ref: typing.Optional\[str\] = Nonedpo\_beta: float = 0.1max\_prompt\_length: int = 128max\_completion\_length: typing.Optional\[int\] = Noneprompt\_text\_column: typing.Optional\[str\] = Nonetext\_column: str = 'text'rejected\_text\_column: typing.Optional\[str\] = Nonepush\_to\_hub: bool = Falseusername: typing.Optional\[str\] = Nonetoken: typing.Optional\[str\] = Noneunsloth: bool = Falsedistributed\_backend: typing.Optional\[str\] = None)

Expand 48 parameters

Parameters

- **model** (str) — Model name to be used for training. Default is “gpt2”.
- **project\_name** (str) — Name of the project and output directory. Default is “project-name”.
- **data\_path** (str) — Path to the dataset. Default is “data”.
- **train\_split** (str) — Configuration for the training data split. Default is “train”.
- **valid\_split** (Optional\[str\]) — Configuration for the validation data split. Default is None.
- **add\_eos\_token** (bool) — Whether to add an EOS token at the end of sequences. Default is True.
- **block\_size** (Union\[int, List\[int\]\]) — Size of the blocks for training, can be a single integer or a list of integers. Default is -1.
- **model\_max\_length** (int) — Maximum length of the model input. Default is 2048.
- **padding** (Optional\[str\]) — Side on which to pad sequences (left or right). Default is “right”.
- **trainer** (str) — Type of trainer to use. Default is “default”.
- **use\_flash\_attention\_2** (bool) — Whether to use flash attention version 2. Default is False.
- **log** (str) — Logging method for experiment tracking. Default is “none”.
- **disable\_gradient\_checkpointing** (bool) — Whether to disable gradient checkpointing. Default is False.
- **logging\_steps** (int) — Number of steps between logging events. Default is -1.
- **eval\_strategy** (str) — Strategy for evaluation (e.g., ‘epoch’). Default is “epoch”.
- **save\_total\_limit** (int) — Maximum number of checkpoints to keep. Default is 1.
- **auto\_find\_batch\_size** (bool) — Whether to automatically find the optimal batch size. Default is False.
- **mixed\_precision** (Optional\[str\]) — Type of mixed precision to use (e.g., ‘fp16’, ‘bf16’, or None). Default is None.
- **lr** (float) — Learning rate for training. Default is 3e-5.
- **epochs** (int) — Number of training epochs. Default is 1.
- **batch\_size** (int) — Batch size for training. Default is 2.
- **warmup\_ratio** (float) — Proportion of training to perform learning rate warmup. Default is 0.1.
- **gradient\_accumulation** (int) — Number of steps to accumulate gradients before updating. Default is 4.
- **optimizer** (str) — Optimizer to use for training. Default is “adamw\_torch”.
- **scheduler** (str) — Learning rate scheduler to use. Default is “linear”.
- **weight\_decay** (float) — Weight decay to apply to the optimizer. Default is 0.0.
- **max\_grad\_norm** (float) — Maximum norm for gradient clipping. Default is 1.0.
- **seed** (int) — Random seed for reproducibility. Default is 42.
- **chat\_template** (Optional\[str\]) — Template for chat-based models, options include: None, zephyr, chatml, or tokenizer. Default is None.
- **quantization** (Optional\[str\]) — Quantization method to use (e.g., ‘int4’, ‘int8’, or None). Default is “int4”.
- **target\_modules** (Optional\[str\]) — Target modules for quantization or fine-tuning. Default is “all-linear”.
- **merge\_adapter** (bool) — Whether to merge the adapter layers. Default is False.
- **peft** (bool) — Whether to use Parameter-Efficient Fine-Tuning (PEFT). Default is False.
- **lora\_r** (int) — Rank of the LoRA matrices. Default is 16.
- **lora\_alpha** (int) — Alpha parameter for LoRA. Default is 32.
- **lora\_dropout** (float) — Dropout rate for LoRA. Default is 0.05.
- **model\_ref** (Optional\[str\]) — Reference model for DPO trainer. Default is None.
- **dpo\_beta** (float) — Beta parameter for DPO trainer. Default is 0.1.
- **max\_prompt\_length** (int) — Maximum length of the prompt. Default is 128.
- **max\_completion\_length** (Optional\[int\]) — Maximum length of the completion. Default is None.
- **prompt\_text\_column** (Optional\[str\]) — Column name for the prompt text. Default is None.
- **text\_column** (str) — Column name for the text data. Default is “text”.
- **rejected\_text\_column** (Optional\[str\]) — Column name for the rejected text data. Default is None.
- **push\_to\_hub** (bool) — Whether to push the model to the Hugging Face Hub. Default is False.
- **username** (Optional\[str\]) — Hugging Face username for authentication. Default is None.
- **token** (Optional\[str\]) — Hugging Face token for authentication. Default is None.
- **unsloth** (bool) — Whether to use the unsloth library. Default is False.
- **distributed\_backend** (Optional\[str\]) — Backend to use for distributed training. Default is None.

LLMTrainingParams: Parameters for training a language model using the autotrain library.

### classautotrain.trainers.sent\_transformers.params.SentenceTransformersParams

[<source>](https://github.com/huggingface/autotrain-advanced/blob/main/src/autotrain/trainers/sent_transformers/params.py#L8)

(data\_path: str = Nonemodel: str = 'microsoft/mpnet-base'lr: float = 3e-05epochs: int = 3max\_seq\_length: int = 128batch\_size: int = 8warmup\_ratio: float = 0.1gradient\_accumulation: int = 1optimizer: str = 'adamw\_torch'scheduler: str = 'linear'weight\_decay: float = 0.0max\_grad\_norm: float = 1.0seed: int = 42train\_split: str = 'train'valid\_split: typing.Optional\[str\] = Nonelogging\_steps: int = -1project\_name: str = 'project-name'auto\_find\_batch\_size: bool = Falsemixed\_precision: typing.Optional\[str\] = Nonesave\_total\_limit: int = 1token: typing.Optional\[str\] = Nonepush\_to\_hub: bool = Falseeval\_strategy: str = 'epoch'username: typing.Optional\[str\] = Nonelog: str = 'none'early\_stopping\_patience: int = 5early\_stopping\_threshold: float = 0.01trainer: str = 'pair\_score'sentence1\_column: str = 'sentence1'sentence2\_column: str = 'sentence2'sentence3\_column: typing.Optional\[str\] = Nonetarget\_column: typing.Optional\[str\] = None)

Expand 32 parameters

Parameters

- **data\_path** (str) — Path to the dataset.
- **model** (str) — Name of the pre-trained model to use. Default is “microsoft/mpnet-base”.
- **lr** (float) — Learning rate for training. Default is 3e-5.
- **epochs** (int) — Number of training epochs. Default is 3.
- **max\_seq\_length** (int) — Maximum sequence length for the input. Default is 128.
- **batch\_size** (int) — Batch size for training. Default is 8.
- **warmup\_ratio** (float) — Proportion of training to perform learning rate warmup. Default is 0.1.
- **gradient\_accumulation** (int) — Number of steps to accumulate gradients before updating. Default is 1.
- **optimizer** (str) — Optimizer to use. Default is “adamw\_torch”.
- **scheduler** (str) — Learning rate scheduler to use. Default is “linear”.
- **weight\_decay** (float) — Weight decay to apply. Default is 0.0.
- **max\_grad\_norm** (float) — Maximum gradient norm for clipping. Default is 1.0.
- **seed** (int) — Random seed for reproducibility. Default is 42.
- **train\_split** (str) — Name of the training data split. Default is “train”.
- **valid\_split** (Optional\[str\]) — Name of the validation data split. Default is None.
- **logging\_steps** (int) — Number of steps between logging. Default is -1.
- **project\_name** (str) — Name of the project for output directory. Default is “project-name”.
- **auto\_find\_batch\_size** (bool) — Whether to automatically find the optimal batch size. Default is False.
- **mixed\_precision** (Optional\[str\]) — Mixed precision training mode (fp16, bf16, or None). Default is None.
- **save\_total\_limit** (int) — Maximum number of checkpoints to save. Default is 1.
- **token** (Optional\[str\]) — Token for accessing Hugging Face Hub. Default is None.
- **push\_to\_hub** (bool) — Whether to push the model to Hugging Face Hub. Default is False.
- **eval\_strategy** (str) — Evaluation strategy to use. Default is “epoch”.
- **username** (Optional\[str\]) — Hugging Face username. Default is None.
- **log** (str) — Logging method for experiment tracking. Default is “none”.
- **early\_stopping\_patience** (int) — Number of epochs with no improvement after which training will be stopped. Default is 5.
- **early\_stopping\_threshold** (float) — Threshold for measuring the new optimum, to qualify as an improvement. Default is 0.01.
- **trainer** (str) — Name of the trainer to use. Default is “pair\_score”.
- **sentence1\_column** (str) — Name of the column containing the first sentence. Default is “sentence1”.
- **sentence2\_column** (str) — Name of the column containing the second sentence. Default is “sentence2”.
- **sentence3\_column** (Optional\[str\]) — Name of the column containing the third sentence (if applicable). Default is None.
- **target\_column** (Optional\[str\]) — Name of the column containing the target variable. Default is None.

SentenceTransformersParams is a configuration class for setting up parameters for training sentence transformers.

### classautotrain.trainers.seq2seq.params.Seq2SeqParams

[<source>](https://github.com/huggingface/autotrain-advanced/blob/main/src/autotrain/trainers/seq2seq/params.py#L8)

(data\_path: str = Nonemodel: str = 'google/flan-t5-base'username: typing.Optional\[str\] = Noneseed: int = 42train\_split: str = 'train'valid\_split: typing.Optional\[str\] = Noneproject\_name: str = 'project-name'token: typing.Optional\[str\] = Nonepush\_to\_hub: bool = Falsetext\_column: str = 'text'target\_column: str = 'target'lr: float = 5e-05epochs: int = 3max\_seq\_length: int = 128max\_target\_length: int = 128batch\_size: int = 2warmup\_ratio: float = 0.1gradient\_accumulation: int = 1optimizer: str = 'adamw\_torch'scheduler: str = 'linear'weight\_decay: float = 0.0max\_grad\_norm: float = 1.0logging\_steps: int = -1eval\_strategy: str = 'epoch'auto\_find\_batch\_size: bool = Falsemixed\_precision: typing.Optional\[str\] = Nonesave\_total\_limit: int = 1peft: bool = Falsequantization: typing.Optional\[str\] = 'int8'lora\_r: int = 16lora\_alpha: int = 32lora\_dropout: float = 0.05target\_modules: str = 'all-linear'log: str = 'none'early\_stopping\_patience: int = 5early\_stopping\_threshold: float = 0.01)

Expand 36 parameters

Parameters

- **data\_path** (str) — Path to the dataset.
- **model** (str) — Name of the model to be used. Default is “google/flan-t5-base”.
- **username** (Optional\[str\]) — Hugging Face Username.
- **seed** (int) — Random seed for reproducibility. Default is 42.
- **train\_split** (str) — Name of the training data split. Default is “train”.
- **valid\_split** (Optional\[str\]) — Name of the validation data split.
- **project\_name** (str) — Name of the project or output directory. Default is “project-name”.
- **token** (Optional\[str\]) — Hub Token for authentication.
- **push\_to\_hub** (bool) — Whether to push the model to the Hugging Face Hub. Default is False.
- **text\_column** (str) — Name of the text column in the dataset. Default is “text”.
- **target\_column** (str) — Name of the target text column in the dataset. Default is “target”.
- **lr** (float) — Learning rate for training. Default is 5e-5.
- **epochs** (int) — Number of training epochs. Default is 3.
- **max\_seq\_length** (int) — Maximum sequence length for input text. Default is 128.
- **max\_target\_length** (int) — Maximum sequence length for target text. Default is 128.
- **batch\_size** (int) — Training batch size. Default is 2.
- **warmup\_ratio** (float) — Proportion of warmup steps. Default is 0.1.
- **gradient\_accumulation** (int) — Number of gradient accumulation steps. Default is 1.
- **optimizer** (str) — Optimizer to be used. Default is “adamw\_torch”.
- **scheduler** (str) — Learning rate scheduler to be used. Default is “linear”.
- **weight\_decay** (float) — Weight decay for the optimizer. Default is 0.0.
- **max\_grad\_norm** (float) — Maximum gradient norm for clipping. Default is 1.0.
- **logging\_steps** (int) — Number of steps between logging. Default is -1 (disabled).
- **eval\_strategy** (str) — Evaluation strategy. Default is “epoch”.
- **auto\_find\_batch\_size** (bool) — Whether to automatically find the batch size. Default is False.
- **mixed\_precision** (Optional\[str\]) — Mixed precision training mode (fp16, bf16, or None).
- **save\_total\_limit** (int) — Maximum number of checkpoints to save. Default is 1.
- **peft** (bool) — Whether to use Parameter-Efficient Fine-Tuning (PEFT). Default is False.
- **quantization** (Optional\[str\]) — Quantization mode (int4, int8, or None). Default is “int8”.
- **lora\_r** (int) — LoRA-R parameter for PEFT. Default is 16.
- **lora\_alpha** (int) — LoRA-Alpha parameter for PEFT. Default is 32.
- **lora\_dropout** (float) — LoRA-Dropout parameter for PEFT. Default is 0.05.
- **target\_modules** (str) — Target modules for PEFT. Default is “all-linear”.
- **log** (str) — Logging method for experiment tracking. Default is “none”.
- **early\_stopping\_patience** (int) — Patience for early stopping. Default is 5.
- **early\_stopping\_threshold** (float) — Threshold for early stopping. Default is 0.01.

Seq2SeqParams is a configuration class for sequence-to-sequence training parameters.

### classautotrain.trainers.token\_classification.params.TokenClassificationParams

[<source>](https://github.com/huggingface/autotrain-advanced/blob/main/src/autotrain/trainers/token_classification/params.py#L8)

(data\_path: str = Nonemodel: str = 'bert-base-uncased'lr: float = 5e-05epochs: int = 3max\_seq\_length: int = 128batch\_size: int = 8warmup\_ratio: float = 0.1gradient\_accumulation: int = 1optimizer: str = 'adamw\_torch'scheduler: str = 'linear'weight\_decay: float = 0.0max\_grad\_norm: float = 1.0seed: int = 42train\_split: str = 'train'valid\_split: typing.Optional\[str\] = Nonetokens\_column: str = 'tokens'tags\_column: str = 'tags'logging\_steps: int = -1project\_name: str = 'project-name'auto\_find\_batch\_size: bool = Falsemixed\_precision: typing.Optional\[str\] = Nonesave\_total\_limit: int = 1token: typing.Optional\[str\] = Nonepush\_to\_hub: bool = Falseeval\_strategy: str = 'epoch'username: typing.Optional\[str\] = Nonelog: str = 'none'early\_stopping\_patience: int = 5early\_stopping\_threshold: float = 0.01)

Expand 29 parameters

Parameters

- **data\_path** (str) — Path to the dataset.
- **model** (str) — Name of the model to use. Default is “bert-base-uncased”.
- **lr** (float) — Learning rate. Default is 5e-5.
- **epochs** (int) — Number of training epochs. Default is 3.
- **max\_seq\_length** (int) — Maximum sequence length. Default is 128.
- **batch\_size** (int) — Training batch size. Default is 8.
- **warmup\_ratio** (float) — Warmup proportion. Default is 0.1.
- **gradient\_accumulation** (int) — Gradient accumulation steps. Default is 1.
- **optimizer** (str) — Optimizer to use. Default is “adamw\_torch”.
- **scheduler** (str) — Scheduler to use. Default is “linear”.
- **weight\_decay** (float) — Weight decay. Default is 0.0.
- **max\_grad\_norm** (float) — Maximum gradient norm. Default is 1.0.
- **seed** (int) — Random seed. Default is 42.
- **train\_split** (str) — Name of the training split. Default is “train”.
- **valid\_split** (Optional\[str\]) — Name of the validation split. Default is None.
- **tokens\_column** (str) — Name of the tokens column. Default is “tokens”.
- **tags\_column** (str) — Name of the tags column. Default is “tags”.
- **logging\_steps** (int) — Number of steps between logging. Default is -1.
- **project\_name** (str) — Name of the project. Default is “project-name”.
- **auto\_find\_batch\_size** (bool) — Whether to automatically find the batch size. Default is False.
- **mixed\_precision** (Optional\[str\]) — Mixed precision setting (fp16, bf16, or None). Default is None.
- **save\_total\_limit** (int) — Total number of checkpoints to save. Default is 1.
- **token** (Optional\[str\]) — Hub token for authentication. Default is None.
- **push\_to\_hub** (bool) — Whether to push the model to the Hugging Face hub. Default is False.
- **eval\_strategy** (str) — Evaluation strategy. Default is “epoch”.
- **username** (Optional\[str\]) — Hugging Face username. Default is None.
- **log** (str) — Logging method for experiment tracking. Default is “none”.
- **early\_stopping\_patience** (int) — Patience for early stopping. Default is 5.
- **early\_stopping\_threshold** (float) — Threshold for early stopping. Default is 0.01.

TokenClassificationParams is a configuration class for token classification training parameters.

### classautotrain.trainers.extractive\_question\_answering.params.ExtractiveQuestionAnsweringParams

[<source>](https://github.com/huggingface/autotrain-advanced/blob/main/src/autotrain/trainers/extractive_question_answering/params.py#L8)

(data\_path: str = Nonemodel: str = 'bert-base-uncased'lr: float = 5e-05epochs: int = 3max\_seq\_length: int = 128max\_doc\_stride: int = 128batch\_size: int = 8warmup\_ratio: float = 0.1gradient\_accumulation: int = 1optimizer: str = 'adamw\_torch'scheduler: str = 'linear'weight\_decay: float = 0.0max\_grad\_norm: float = 1.0seed: int = 42train\_split: str = 'train'valid\_split: typing.Optional\[str\] = Nonetext\_column: str = 'context'question\_column: str = 'question'answer\_column: str = 'answers'logging\_steps: int = -1project\_name: str = 'project-name'auto\_find\_batch\_size: bool = Falsemixed\_precision: typing.Optional\[str\] = Nonesave\_total\_limit: int = 1token: typing.Optional\[str\] = Nonepush\_to\_hub: bool = Falseeval\_strategy: str = 'epoch'username: typing.Optional\[str\] = Nonelog: str = 'none'early\_stopping\_patience: int = 5early\_stopping\_threshold: float = 0.01)

Expand 31 parameters

Parameters

- **data\_path** (str) — Path to the dataset.
- **model** (str) — Pre-trained model name. Default is “bert-base-uncased”.
- **lr** (float) — Learning rate for the optimizer. Default is 5e-5.
- **epochs** (int) — Number of training epochs. Default is 3.
- **max\_seq\_length** (int) — Maximum sequence length for inputs. Default is 128.
- **max\_doc\_stride** (int) — Maximum document stride for splitting context. Default is 128.
- **batch\_size** (int) — Batch size for training. Default is 8.
- **warmup\_ratio** (float) — Warmup proportion for learning rate scheduler. Default is 0.1.
- **gradient\_accumulation** (int) — Number of gradient accumulation steps. Default is 1.
- **optimizer** (str) — Optimizer type. Default is “adamw\_torch”.
- **scheduler** (str) — Learning rate scheduler type. Default is “linear”.
- **weight\_decay** (float) — Weight decay for the optimizer. Default is 0.0.
- **max\_grad\_norm** (float) — Maximum gradient norm for clipping. Default is 1.0.
- **seed** (int) — Random seed for reproducibility. Default is 42.
- **train\_split** (str) — Name of the training data split. Default is “train”.
- **valid\_split** (Optional\[str\]) — Name of the validation data split. Default is None.
- **text\_column** (str) — Column name for context/text. Default is “context”.
- **question\_column** (str) — Column name for questions. Default is “question”.
- **answer\_column** (str) — Column name for answers. Default is “answers”.
- **logging\_steps** (int) — Number of steps between logging. Default is -1.
- **project\_name** (str) — Name of the project for output directory. Default is “project-name”.
- **auto\_find\_batch\_size** (bool) — Automatically find optimal batch size. Default is False.
- **mixed\_precision** (Optional\[str\]) — Mixed precision training mode (fp16, bf16, or None). Default is None.
- **save\_total\_limit** (int) — Maximum number of checkpoints to save. Default is 1.
- **token** (Optional\[str\]) — Authentication token for Hugging Face Hub. Default is None.
- **push\_to\_hub** (bool) — Whether to push the model to Hugging Face Hub. Default is False.
- **eval\_strategy** (str) — Evaluation strategy during training. Default is “epoch”.
- **username** (Optional\[str\]) — Hugging Face username for authentication. Default is None.
- **log** (str) — Logging method for experiment tracking. Default is “none”.
- **early\_stopping\_patience** (int) — Number of epochs with no improvement for early stopping. Default is 5.
- **early\_stopping\_threshold** (float) — Threshold for early stopping improvement. Default is 0.01.

ExtractiveQuestionAnsweringParams

### classautotrain.trainers.text\_classification.params.TextClassificationParams

[<source>](https://github.com/huggingface/autotrain-advanced/blob/main/src/autotrain/trainers/text_classification/params.py#L8)

(data\_path: str = Nonemodel: str = 'bert-base-uncased'lr: float = 5e-05epochs: int = 3max\_seq\_length: int = 128batch\_size: int = 8warmup\_ratio: float = 0.1gradient\_accumulation: int = 1optimizer: str = 'adamw\_torch'scheduler: str = 'linear'weight\_decay: float = 0.0max\_grad\_norm: float = 1.0seed: int = 42train\_split: str = 'train'valid\_split: typing.Optional\[str\] = Nonetext\_column: str = 'text'target\_column: str = 'target'logging\_steps: int = -1project\_name: str = 'project-name'auto\_find\_batch\_size: bool = Falsemixed\_precision: typing.Optional\[str\] = Nonesave\_total\_limit: int = 1token: typing.Optional\[str\] = Nonepush\_to\_hub: bool = Falseeval\_strategy: str = 'epoch'username: typing.Optional\[str\] = Nonelog: str = 'none'early\_stopping\_patience: int = 5early\_stopping\_threshold: float = 0.01)

Expand 29 parameters

Parameters

- **data\_path** (str) — Path to the dataset.
- **model** (str) — Name of the model to use. Default is “bert-base-uncased”.
- **lr** (float) — Learning rate. Default is 5e-5.
- **epochs** (int) — Number of training epochs. Default is 3.
- **max\_seq\_length** (int) — Maximum sequence length. Default is 128.
- **batch\_size** (int) — Training batch size. Default is 8.
- **warmup\_ratio** (float) — Warmup proportion. Default is 0.1.
- **gradient\_accumulation** (int) — Number of gradient accumulation steps. Default is 1.
- **optimizer** (str) — Optimizer to use. Default is “adamw\_torch”.
- **scheduler** (str) — Scheduler to use. Default is “linear”.
- **weight\_decay** (float) — Weight decay. Default is 0.0.
- **max\_grad\_norm** (float) — Maximum gradient norm. Default is 1.0.
- **seed** (int) — Random seed. Default is 42.
- **train\_split** (str) — Name of the training split. Default is “train”.
- **valid\_split** (Optional\[str\]) — Name of the validation split. Default is None.
- **text\_column** (str) — Name of the text column in the dataset. Default is “text”.
- **target\_column** (str) — Name of the target column in the dataset. Default is “target”.
- **logging\_steps** (int) — Number of steps between logging. Default is -1.
- **project\_name** (str) — Name of the project. Default is “project-name”.
- **auto\_find\_batch\_size** (bool) — Whether to automatically find the batch size. Default is False.
- **mixed\_precision** (Optional\[str\]) — Mixed precision setting (fp16, bf16, or None). Default is None.
- **save\_total\_limit** (int) — Total number of checkpoints to save. Default is 1.
- **token** (Optional\[str\]) — Hub token for authentication. Default is None.
- **push\_to\_hub** (bool) — Whether to push the model to the hub. Default is False.
- **eval\_strategy** (str) — Evaluation strategy. Default is “epoch”.
- **username** (Optional\[str\]) — Hugging Face username. Default is None.
- **log** (str) — Logging method for experiment tracking. Default is “none”.
- **early\_stopping\_patience** (int) — Number of epochs with no improvement after which training will be stopped. Default is 5.
- **early\_stopping\_threshold** (float) — Threshold for measuring the new optimum to continue training. Default is 0.01.

`TextClassificationParams` is a configuration class for text classification training parameters.

### classautotrain.trainers.text\_regression.params.TextRegressionParams

[<source>](https://github.com/huggingface/autotrain-advanced/blob/main/src/autotrain/trainers/text_regression/params.py#L8)

(data\_path: str = Nonemodel: str = 'bert-base-uncased'lr: float = 5e-05epochs: int = 3max\_seq\_length: int = 128batch\_size: int = 8warmup\_ratio: float = 0.1gradient\_accumulation: int = 1optimizer: str = 'adamw\_torch'scheduler: str = 'linear'weight\_decay: float = 0.0max\_grad\_norm: float = 1.0seed: int = 42train\_split: str = 'train'valid\_split: typing.Optional\[str\] = Nonetext\_column: str = 'text'target\_column: str = 'target'logging\_steps: int = -1project\_name: str = 'project-name'auto\_find\_batch\_size: bool = Falsemixed\_precision: typing.Optional\[str\] = Nonesave\_total\_limit: int = 1token: typing.Optional\[str\] = Nonepush\_to\_hub: bool = Falseeval\_strategy: str = 'epoch'username: typing.Optional\[str\] = Nonelog: str = 'none'early\_stopping\_patience: int = 5early\_stopping\_threshold: float = 0.01)

Expand 29 parameters

Parameters

- **data\_path** (str) — Path to the dataset.
- **model** (str) — Name of the pre-trained model to use. Default is “bert-base-uncased”.
- **lr** (float) — Learning rate for the optimizer. Default is 5e-5.
- **epochs** (int) — Number of training epochs. Default is 3.
- **max\_seq\_length** (int) — Maximum sequence length for the inputs. Default is 128.
- **batch\_size** (int) — Batch size for training. Default is 8.
- **warmup\_ratio** (float) — Proportion of training to perform learning rate warmup. Default is 0.1.
- **gradient\_accumulation** (int) — Number of steps to accumulate gradients before updating. Default is 1.
- **optimizer** (str) — Optimizer to use. Default is “adamw\_torch”.
- **scheduler** (str) — Learning rate scheduler to use. Default is “linear”.
- **weight\_decay** (float) — Weight decay to apply. Default is 0.0.
- **max\_grad\_norm** (float) — Maximum norm for the gradients. Default is 1.0.
- **seed** (int) — Random seed for reproducibility. Default is 42.
- **train\_split** (str) — Name of the training data split. Default is “train”.
- **valid\_split** (Optional\[str\]) — Name of the validation data split. Default is None.
- **text\_column** (str) — Name of the column containing text data. Default is “text”.
- **target\_column** (str) — Name of the column containing target data. Default is “target”.
- **logging\_steps** (int) — Number of steps between logging. Default is -1 (no logging).
- **project\_name** (str) — Name of the project for output directory. Default is “project-name”.
- **auto\_find\_batch\_size** (bool) — Whether to automatically find the batch size. Default is False.
- **mixed\_precision** (Optional\[str\]) — Mixed precision training mode (fp16, bf16, or None). Default is None.
- **save\_total\_limit** (int) — Maximum number of checkpoints to save. Default is 1.
- **token** (Optional\[str\]) — Token for accessing Hugging Face Hub. Default is None.
- **push\_to\_hub** (bool) — Whether to push the model to Hugging Face Hub. Default is False.
- **eval\_strategy** (str) — Evaluation strategy to use. Default is “epoch”.
- **username** (Optional\[str\]) — Hugging Face username. Default is None.
- **log** (str) — Logging method for experiment tracking. Default is “none”.
- **early\_stopping\_patience** (int) — Number of epochs with no improvement after which training will be stopped. Default is 5.
- **early\_stopping\_threshold** (float) — Threshold for measuring the new optimum, to qualify as an improvement. Default is 0.01.

TextRegressionParams is a configuration class for setting up text regression training parameters.

### Image Tasks

### classautotrain.trainers.image\_classification.params.ImageClassificationParams

[<source>](https://github.com/huggingface/autotrain-advanced/blob/main/src/autotrain/trainers/image_classification/params.py#L8)

(data\_path: str = Nonemodel: str = 'google/vit-base-patch16-224'username: typing.Optional\[str\] = Nonelr: float = 5e-05epochs: int = 3batch\_size: int = 8warmup\_ratio: float = 0.1gradient\_accumulation: int = 1optimizer: str = 'adamw\_torch'scheduler: str = 'linear'weight\_decay: float = 0.0max\_grad\_norm: float = 1.0seed: int = 42train\_split: str = 'train'valid\_split: typing.Optional\[str\] = Nonelogging\_steps: int = -1project\_name: str = 'project-name'auto\_find\_batch\_size: bool = Falsemixed\_precision: typing.Optional\[str\] = Nonesave\_total\_limit: int = 1token: typing.Optional\[str\] = Nonepush\_to\_hub: bool = Falseeval\_strategy: str = 'epoch'image\_column: str = 'image'target\_column: str = 'target'log: str = 'none'early\_stopping\_patience: int = 5early\_stopping\_threshold: float = 0.01)

Expand 28 parameters

Parameters

- **data\_path** (str) — Path to the dataset.
- **model** (str) — Pre-trained model name or path. Default is “google/vit-base-patch16-224”.
- **username** (Optional\[str\]) — Hugging Face account username.
- **lr** (float) — Learning rate for the optimizer. Default is 5e-5.
- **epochs** (int) — Number of epochs for training. Default is 3.
- **batch\_size** (int) — Batch size for training. Default is 8.
- **warmup\_ratio** (float) — Warmup ratio for learning rate scheduler. Default is 0.1.
- **gradient\_accumulation** (int) — Number of gradient accumulation steps. Default is 1.
- **optimizer** (str) — Optimizer type. Default is “adamw\_torch”.
- **scheduler** (str) — Learning rate scheduler type. Default is “linear”.
- **weight\_decay** (float) — Weight decay for the optimizer. Default is 0.0.
- **max\_grad\_norm** (float) — Maximum gradient norm for clipping. Default is 1.0.
- **seed** (int) — Random seed for reproducibility. Default is 42.
- **train\_split** (str) — Name of the training data split. Default is “train”.
- **valid\_split** (Optional\[str\]) — Name of the validation data split.
- **logging\_steps** (int) — Number of steps between logging. Default is -1.
- **project\_name** (str) — Name of the project for output directory. Default is “project-name”.
- **auto\_find\_batch\_size** (bool) — Automatically find optimal batch size. Default is False.
- **mixed\_precision** (Optional\[str\]) — Mixed precision training mode (fp16, bf16, or None).
- **save\_total\_limit** (int) — Maximum number of checkpoints to keep. Default is 1.
- **token** (Optional\[str\]) — Hugging Face Hub token for authentication.
- **push\_to\_hub** (bool) — Whether to push the model to Hugging Face Hub. Default is False.
- **eval\_strategy** (str) — Evaluation strategy during training. Default is “epoch”.
- **image\_column** (str) — Column name for images in the dataset. Default is “image”.
- **target\_column** (str) — Column name for target labels in the dataset. Default is “target”.
- **log** (str) — Logging method for experiment tracking. Default is “none”.
- **early\_stopping\_patience** (int) — Number of epochs with no improvement for early stopping. Default is 5.
- **early\_stopping\_threshold** (float) — Threshold for early stopping. Default is 0.01.

ImageClassificationParams is a configuration class for image classification training parameters.

### classautotrain.trainers.image\_regression.params.ImageRegressionParams

[<source>](https://github.com/huggingface/autotrain-advanced/blob/main/src/autotrain/trainers/image_regression/params.py#L8)

(data\_path: str = Nonemodel: str = 'google/vit-base-patch16-224'username: typing.Optional\[str\] = Nonelr: float = 5e-05epochs: int = 3batch\_size: int = 8warmup\_ratio: float = 0.1gradient\_accumulation: int = 1optimizer: str = 'adamw\_torch'scheduler: str = 'linear'weight\_decay: float = 0.0max\_grad\_norm: float = 1.0seed: int = 42train\_split: str = 'train'valid\_split: typing.Optional\[str\] = Nonelogging\_steps: int = -1project\_name: str = 'project-name'auto\_find\_batch\_size: bool = Falsemixed\_precision: typing.Optional\[str\] = Nonesave\_total\_limit: int = 1token: typing.Optional\[str\] = Nonepush\_to\_hub: bool = Falseeval\_strategy: str = 'epoch'image\_column: str = 'image'target\_column: str = 'target'log: str = 'none'early\_stopping\_patience: int = 5early\_stopping\_threshold: float = 0.01)

Expand 28 parameters

Parameters

- **data\_path** (str) — Path to the dataset.
- **model** (str) — Name of the model to use. Default is “google/vit-base-patch16-224”.
- **username** (Optional\[str\]) — Hugging Face Username.
- **lr** (float) — Learning rate. Default is 5e-5.
- **epochs** (int) — Number of training epochs. Default is 3.
- **batch\_size** (int) — Training batch size. Default is 8.
- **warmup\_ratio** (float) — Warmup proportion. Default is 0.1.
- **gradient\_accumulation** (int) — Gradient accumulation steps. Default is 1.
- **optimizer** (str) — Optimizer to use. Default is “adamw\_torch”.
- **scheduler** (str) — Scheduler to use. Default is “linear”.
- **weight\_decay** (float) — Weight decay. Default is 0.0.
- **max\_grad\_norm** (float) — Max gradient norm. Default is 1.0.
- **seed** (int) — Random seed. Default is 42.
- **train\_split** (str) — Train split name. Default is “train”.
- **valid\_split** (Optional\[str\]) — Validation split name.
- **logging\_steps** (int) — Logging steps. Default is -1.
- **project\_name** (str) — Output directory name. Default is “project-name”.
- **auto\_find\_batch\_size** (bool) — Whether to auto find batch size. Default is False.
- **mixed\_precision** (Optional\[str\]) — Mixed precision type (fp16, bf16, or None).
- **save\_total\_limit** (int) — Save total limit. Default is 1.
- **token** (Optional\[str\]) — Hub Token.
- **push\_to\_hub** (bool) — Whether to push to hub. Default is False.
- **eval\_strategy** (str) — Evaluation strategy. Default is “epoch”.
- **image\_column** (str) — Image column name. Default is “image”.
- **target\_column** (str) — Target column name. Default is “target”.
- **log** (str) — Logging using experiment tracking. Default is “none”.
- **early\_stopping\_patience** (int) — Early stopping patience. Default is 5.
- **early\_stopping\_threshold** (float) — Early stopping threshold. Default is 0.01.

ImageRegressionParams is a configuration class for image regression training parameters.

### classautotrain.trainers.object\_detection.params.ObjectDetectionParams

[<source>](https://github.com/huggingface/autotrain-advanced/blob/main/src/autotrain/trainers/object_detection/params.py#L8)

(data\_path: str = Nonemodel: str = 'google/vit-base-patch16-224'username: typing.Optional\[str\] = Nonelr: float = 5e-05epochs: int = 3batch\_size: int = 8warmup\_ratio: float = 0.1gradient\_accumulation: int = 1optimizer: str = 'adamw\_torch'scheduler: str = 'linear'weight\_decay: float = 0.0max\_grad\_norm: float = 1.0seed: int = 42train\_split: str = 'train'valid\_split: typing.Optional\[str\] = Nonelogging\_steps: int = -1project\_name: str = 'project-name'auto\_find\_batch\_size: bool = Falsemixed\_precision: typing.Optional\[str\] = Nonesave\_total\_limit: int = 1token: typing.Optional\[str\] = Nonepush\_to\_hub: bool = Falseeval\_strategy: str = 'epoch'image\_column: str = 'image'objects\_column: str = 'objects'log: str = 'none'image\_square\_size: typing.Optional\[int\] = 600early\_stopping\_patience: int = 5early\_stopping\_threshold: float = 0.01)

Expand 29 parameters

Parameters

- **data\_path** (str) — Path to the dataset.
- **model** (str) — Name of the model to be used. Default is “google/vit-base-patch16-224”.
- **username** (Optional\[str\]) — Hugging Face Username.
- **lr** (float) — Learning rate. Default is 5e-5.
- **epochs** (int) — Number of training epochs. Default is 3.
- **batch\_size** (int) — Training batch size. Default is 8.
- **warmup\_ratio** (float) — Warmup proportion. Default is 0.1.
- **gradient\_accumulation** (int) — Gradient accumulation steps. Default is 1.
- **optimizer** (str) — Optimizer to be used. Default is “adamw\_torch”.
- **scheduler** (str) — Scheduler to be used. Default is “linear”.
- **weight\_decay** (float) — Weight decay. Default is 0.0.
- **max\_grad\_norm** (float) — Max gradient norm. Default is 1.0.
- **seed** (int) — Random seed. Default is 42.
- **train\_split** (str) — Name of the training data split. Default is “train”.
- **valid\_split** (Optional\[str\]) — Name of the validation data split.
- **logging\_steps** (int) — Number of steps between logging. Default is -1.
- **project\_name** (str) — Name of the project for output directory. Default is “project-name”.
- **auto\_find\_batch\_size** (bool) — Whether to automatically find batch size. Default is False.
- **mixed\_precision** (Optional\[str\]) — Mixed precision type (fp16, bf16, or None).
- **save\_total\_limit** (int) — Total number of checkpoints to save. Default is 1.
- **token** (Optional\[str\]) — Hub Token for authentication.
- **push\_to\_hub** (bool) — Whether to push the model to the Hugging Face Hub. Default is False.
- **eval\_strategy** (str) — Evaluation strategy. Default is “epoch”.
- **image\_column** (str) — Name of the image column in the dataset. Default is “image”.
- **objects\_column** (str) — Name of the target column in the dataset. Default is “objects”.
- **log** (str) — Logging method for experiment tracking. Default is “none”.
- **image\_square\_size** (Optional\[int\]) — Longest size to which the image will be resized, then padded to square. Default is 600.
- **early\_stopping\_patience** (int) — Number of epochs with no improvement after which training will be stopped. Default is 5.
- **early\_stopping\_threshold** (float) — Minimum change to qualify as an improvement. Default is 0.01.

ObjectDetectionParams is a configuration class for object detection training parameters.

### Tabular Tasks

### classautotrain.trainers.tabular.params.TabularParams

[<source>](https://github.com/huggingface/autotrain-advanced/blob/main/src/autotrain/trainers/tabular/params.py#L8)

(data\_path: str = Nonemodel: str = 'xgboost'username: typing.Optional\[str\] = Noneseed: int = 42train\_split: str = 'train'valid\_split: typing.Optional\[str\] = Noneproject\_name: str = 'project-name'token: typing.Optional\[str\] = Nonepush\_to\_hub: bool = Falseid\_column: str = 'id'target\_columns: typing.Union\[typing.List\[str\], str\] = \['target'\]categorical\_columns: typing.Optional\[typing.List\[str\]\] = Nonenumerical\_columns: typing.Optional\[typing.List\[str\]\] = Nonetask: str = 'classification'num\_trials: int = 10time\_limit: int = 600categorical\_imputer: typing.Optional\[str\] = Nonenumerical\_imputer: typing.Optional\[str\] = Nonenumeric\_scaler: typing.Optional\[str\] = None)

Parameters

- **data\_path** (str) — Path to the dataset.
- **model** (str) — Name of the model to use. Default is “xgboost”.
- **username** (Optional\[str\]) — Hugging Face Username.
- **seed** (int) — Random seed for reproducibility. Default is 42.
- **train\_split** (str) — Name of the training data split. Default is “train”.
- **valid\_split** (Optional\[str\]) — Name of the validation data split.
- **project\_name** (str) — Name of the output directory. Default is “project-name”.
- **token** (Optional\[str\]) — Hub Token for authentication.
- **push\_to\_hub** (bool) — Whether to push the model to the hub. Default is False.
- **id\_column** (str) — Name of the ID column. Default is “id”.
- **target\_columns** (Union\[List\[str\], str\]) — Target column(s) in the dataset. Default is \[“target”\].
- **categorical\_columns** (Optional\[List\[str\]\]) — List of categorical columns.
- **numerical\_columns** (Optional\[List\[str\]\]) — List of numerical columns.
- **task** (str) — Type of task (e.g., “classification”). Default is “classification”.
- **num\_trials** (int) — Number of trials for hyperparameter optimization. Default is 10.
- **time\_limit** (int) — Time limit for training in seconds. Default is 600.
- **categorical\_imputer** (Optional\[str\]) — Imputer strategy for categorical columns.
- **numerical\_imputer** (Optional\[str\]) — Imputer strategy for numerical columns.
- **numeric\_scaler** (Optional\[str\]) — Scaler strategy for numerical columns.

TabularParams is a configuration class for tabular data training parameters.

[<>Update on GitHub](https://github.com/huggingface/autotrain-advanced/blob/main/docs/source/quickstart_py.mdx)

[←Train on Spaces](https://huggingface.co/docs/autotrain/en/quickstart_spaces) [Train Locally→](https://huggingface.co/docs/autotrain/en/quickstart)