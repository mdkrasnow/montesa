AutoTrain documentation

LLM Finetuning with AutoTrain Advanced

# AutoTrain

🏡 View all docsAWS Trainium & InferentiaAccelerateAmazon SageMakerArgillaAutoTrainBitsandbytesChat UIDataset viewerDatasetsDiffusersDistilabelEvaluateGradioHubHub Python LibraryHuggingface.jsInference Endpoints (dedicated)Inference ProvidersLeaderboardsLightevalOptimumPEFTSafetensorsSentence TransformersTRLTasksText Embeddings InferenceText Generation InferenceTokenizersTransformersTransformers.jssmolagentstimm

Search documentation
`Ctrl+K`

mainv0.8.24v0.7.129v0.6.48v0.5.2EN

[4,366](https://github.com/huggingface/autotrain-advanced)

You are viewing main version, which requires installation from source. If you'd like
regular pip install, checkout the latest stable version ( [v0.8.24](https://huggingface.co/docs/autotrain/v0.8.24/tasks/llm_finetuning)).


![Hugging Face's logo](https://huggingface.co/front/assets/huggingface_logo-noborder.svg)

Join the Hugging Face community

and get access to the augmented documentation experience


Collaborate on models, datasets and Spaces


Faster examples with accelerated inference


Switch between documentation themes


[Sign Up](https://huggingface.co/join)

to get started

# LLM Finetuning with AutoTrain Advanced

AutoTrain Advanced makes it easy to fine-tune large language models (LLMs) for your specific use cases. This guide covers everything you need to know about LLM fine-tuning.

## Key Features

- Simple data preparation with CSV and JSONL formats
- Support for multiple training approaches (SFT, DPO, ORPO)
- Built-in chat templates
- Local and cloud training options
- Optimized training parameters

## Supported Training Methods

AutoTrain supports multiple specialized trainers:

- `llm`: Generic LLM trainer
- `llm-sft`: Supervised Fine-Tuning trainer
- `llm-reward`: Reward modeling trainer
- `llm-dpo`: Direct Preference Optimization trainer
- `llm-orpo`: ORPO (Optimal Reward Policy Optimization) trainer

## Data Preparation

LLM finetuning accepts data in CSV and JSONL formats. JSONL is the preferred format.
How data is formatted depends on the task you are training the LLM for.

### Classic Text Generation

For text generation, the data should be in the following format:

| text |
| --- |
| wikipedia is a free online encyclopedia |
| it is a collaborative project |
| that anyone can edit |
| wikipedia is the largest and most popular general reference work on the internet |

An example dataset for this format can be found here: [stas/openwebtext-10k](https://huggingface.co/datasets/stas/openwebtext-10k)

Example tasks:

- Text generation
- Code completion

Compatible trainers:

- SFT Trainer
- Generic Trainer

### Chatbot / question-answering / code generation / function calling

For this task, you can use CSV or JSONL data. If you are formatting the data yourself (adding start, end tokens, etc.), you can use CSV or JSONL format.
If you do not want to format the data yourself and want `--chat-template` parameter to format the data for you, you must use JSONL format.
In both cases, CSV and JSONL can be used interchangeably but JSONL is the most preferred format.

To train a chatbot, your data will have `content` and `role`. Some models support `system` role as well.

Here is an example of a chatbot dataset (single sample):

Copied

```
[{'content': 'Help write a letter of 100 -200 words to my future self for '\
             'Kyra, reflecting on her goals and aspirations.',\
  'role': 'user'},\
 {'content': 'Dear Future Self,\n'\
             '\n'\
             "I hope you're happy and proud of what you've achieved. As I "\
             "write this, I'm excited to think about our goals and how far "\
             "you've come. One goal was to be a machine learning engineer. I "\
             "hope you've worked hard and become skilled in this field. Keep "\
             'learning and innovating. Traveling was important to us. I hope '\
             "you've seen different places and enjoyed the beauty of our "\
             'world. Remember the memories and lessons. Starting a family '\
             'mattered to us. If you have kids, treasure every moment. Be '\
             'patient, loving, and grateful for your family.\n'\
             '\n'\
             'Take care of yourself. Rest, reflect, and cherish the time you '\
             'spend with loved ones. Remember your dreams and celebrate what '\
             "you've achieved. Your determination brought you here. I'm "\
             "excited to see the person you've become, the impact you've made, "\
             'and the love and joy in your life. Embrace opportunities and '\
             'keep dreaming big.\n'\
             '\n'\
             'With love,\n'\
             'Kyra',\
  'role': 'assistant'}]
```

As you can see, the data has `content` and `role` columns. The `role` column can be `user` or `assistant` or `system`.
This data is, however, not formatted for training. You can use the `--chat-template` parameter to format the data during training.

`--chat-template` supports the following kinds of templates:

- `none` (default)
- `zephyr`
- `chatml`
- `tokenizer`: use chat template mentioned in tokenizer config

A multi-line sample is also shown below:

Copied

```
[{"content": "hello", "role": "user"}, {"content": "hi nice to meet you", "role": "assistant"}]
[{"content": "how are you", "role": "user"}, {"content": "I am fine", "role": "assistant"}]
[{"content": "What is your name?", "role": "user"}, {"content": "My name is Mary", "role": "assistant"}]
[{"content": "Which is the best programming language?", "role": "user"}, {"content": "Python", "role": "assistant"}]
.
.
.
```

An example dataset for this format can be found here: [HuggingFaceH4/no\_robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots)

If you dont want to format the data using `--chat-template`, you can format the data yourself and use the following format:

Copied

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 03 Oct 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHelp write a letter of 100 -200 words to my future self for Kyra, reflecting on her goals and aspirations.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nDear Future Self,\n\nI hope you're happy and proud of what you've achieved. As I write this, I'm excited to think about our goals and how far you've come. One goal was to be a machine learning engineer. I hope you've worked hard and become skilled in this field. Keep learning and innovating. Traveling was important to us. I hope you've seen different places and enjoyed the beauty of our world. Remember the memories and lessons. Starting a family mattered to us. If you have kids, treasure every moment. Be patient, loving, and grateful for your family.\n\nTake care of yourself. Rest, reflect, and cherish the time you spend with loved ones. Remember your dreams and celebrate what you've achieved. Your determination brought you here. I'm excited to see the person you've become, the impact you've made, and the love and joy in your life. Embrace opportunities and keep dreaming big.\n\nWith love,\nKyra<|eot_id|>
```

A sample multi-line dataset is shown below:

Copied

```
[{"text": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 03 Oct 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nhello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nhi nice to meet you<|eot_id|>"}]
[{"text": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 03 Oct 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nhow are you<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nI am fine<|eot_id|>"}]
[{"text": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 03 Oct 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat is your name?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nMy name is Mary<|eot_id|>"}]
[{"text": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 03 Oct 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhich is the best programming language?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nPython<|eot_id|>"}]
.
.
.
```

An example dataset for this format can be found here: [timdettmers/openassistant-guanaco](https://huggingface.co/datasets/timdettmers/openassistant-guanaco)

In the examples above, we have seen only two turns: one from the user and one from the assistant. However, you can have multiple turns from the user and assistant in a single sample.

Chat models can be trained using the following trainers:

- SFT Trainer:

  - requires only `text` column
  - example dataset: [HuggingFaceH4/no\_robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots)
- Generic Trainer:

  - requires only `text` column
  - example dataset: [HuggingFaceH4/no\_robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots)
- Reward Trainer:

  - requires `text` and `rejected_text` columns
  - example dataset: [trl-lib/ultrafeedback\_binarized](https://huggingface.co/datasets/trl-lib/ultrafeedback_binarized)
- DPO Trainer:

  - requires `prompt`, `text`, and `rejected_text` columns
  - example dataset: [trl-lib/ultrafeedback\_binarized](https://huggingface.co/datasets/trl-lib/ultrafeedback_binarized)
- ORPO Trainer:

  - requires `prompt`, `text`, and `rejected_text` columns
  - example dataset: [trl-lib/ultrafeedback\_binarized](https://huggingface.co/datasets/trl-lib/ultrafeedback_binarized)

The only difference between the data format for reward trainer and DPO/ORPO trainer is that the reward trainer requires only `text` and `rejected_text` columns, while the DPO/ORPO trainer requires an additional `prompt` column.

## Best Practices for LLM Fine-tuning

### Memory Optimization

- Use appropriate `block_size` and `model_max_length` for your hardware
- Enable mixed precision training when possible
- Utilize PEFT techniques for large models

### Data Quality

- Clean and validate your training data
- Ensure balanced conversation samples
- Use appropriate chat templates

### Training Tips

- Start with small learning rates
- Monitor training metrics using tensorboard
- Validate model outputs during training

### Related Resources

- [AutoTrain Documentation](https://huggingface.co/docs/autotrain)
- [Example Fine-tuned Models](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads)
- [Training Datasets](https://huggingface.co/datasets?task_categories=task_categories:text-generation)

## Training

### Local Training

Locally the training can be performed by using `autotrain --config config.yaml` command. The `config.yaml` file should contain the following parameters:

Copied

```
task: llm-orpo
base_model: meta-llama/Meta-Llama-3-8B-Instruct
project_name: autotrain-llama3-8b-orpo
log: tensorboard
backend: local

data:
  path: argilla/distilabel-capybara-dpo-7k-binarized
  train_split: train
  valid_split: null
  chat_template: chatml
  column_mapping:
    text_column: chosen
    rejected_text_column: rejected
    prompt_text_column: prompt

params:
  block_size: 1024
  model_max_length: 8192
  max_prompt_length: 512
  epochs: 3
  batch_size: 2
  lr: 3e-5
  peft: true
  quantization: int4
  target_modules: all-linear
  padding: right
  optimizer: adamw_torch
  scheduler: linear
  gradient_accumulation: 4
  mixed_precision: fp16

hub:
  username: ${HF_USERNAME}
  token: ${HF_TOKEN}
  push_to_hub: true
```

In the above config file, we are training a model using the ORPO trainer.
The model is trained on the `meta-llama/Meta-Llama-3-8B-Instruct` model.
The data is `argilla/distilabel-capybara-dpo-7k-binarized` dataset. The `chat_template` parameter is set to `chatml`.
The `column_mapping` parameter is used to map the columns in the dataset to the required columns for the ORPO trainer.
The `params` section contains the training parameters such as `block_size`, `model_max_length`, `epochs`, `batch_size`, `lr`, `peft`, `quantization`, `target_modules`, `padding`, `optimizer`, `scheduler`, `gradient_accumulation`, and `mixed_precision`.
The `hub` section contains the username and token for the Hugging Face account and the `push_to_hub` parameter is set to `true` to push the trained model to the Hugging Face Hub.

If you have training file locally, you can change data part to:

Copied

```
data:
  path: path/to/training/file
  train_split: train # name of the training file
  valid_split: null
  chat_template: chatml
  column_mapping:
    text_column: chosen
    rejected_text_column: rejected
    prompt_text_column: prompt
```

The above assumes you have `train.csv` or `train.jsonl` in the `path/to/training/file` directory and you will be applying `chatml` template to the data.

You can run the training using the following command:

Copied

```
$ autotrain --config config.yaml
```

More example config files for finetuning different types of lllm and different tasks can be found in the [here](https://github.com/huggingface/autotrain-advanced/tree/main/configs/llm_finetuning).

### Training in Hugging Face Spaces

If you are training in Hugging Face Spaces, everything is the same as local training:

![llm-finetuning](https://raw.githubusercontent.com/huggingface/autotrain-advanced/main/static/llm_orpo_example.png)

In the UI, you need to make sure you select the right model, the dataset and the splits. Special care should be taken for `column_mapping`.

Once you are happy with the parameters, you can click on the `Start Training` button to start the training process.

## Parameters

### LLM Fine Tuning Parameters

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

### Task specific parameters

The length parameters used for different trainers can be different. Some require more context than others.

- block\_size: This is the maximum sequence length or length of one block of text. Setting to -1 determines block size automatically. Default is -1.
- model\_max\_length: Set the maximum length for the model to process in a single batch, which can affect both performance and memory usage. Default is 1024
- max\_prompt\_length: Specify the maximum length for prompts used in training, particularly relevant for tasks requiring initial contextual input. Used only for `orpo` and `dpo` trainer.
- max\_completion\_length: Completion length to use, for orpo: encoder-decoder models only. For dpo, it is the length of the completion text.

**NOTE**:

- block size cannot be greater than model\_max\_length!
- max\_prompt\_length cannot be greater than model\_max\_length!
- max\_prompt\_length cannot be greater than block\_size!
- max\_completion\_length cannot be greater than model\_max\_length!
- max\_completion\_length cannot be greater than block\_size!

**NOTE**: Not following these constraints will result in an error / nan losses.

#### Generic Trainer

Copied

```
--add_eos_token, --add-eos-token
                    Toggle whether to automatically add an End Of Sentence (EOS) token at the end of texts, which can be critical for certain
                    types of models like language models. Only used for `default` trainer
--block_size BLOCK_SIZE, --block-size BLOCK_SIZE
                    Specify the block size for processing sequences. This is maximum sequence length or length of one block of text. Setting to
                    -1 determines block size automatically. Default is -1.
--model_max_length MODEL_MAX_LENGTH, --model-max-length MODEL_MAX_LENGTH
                    Set the maximum length for the model to process in a single batch, which can affect both performance and memory usage.
                    Default is 1024
```

#### SFT Trainer

Copied

```
--block_size BLOCK_SIZE, --block-size BLOCK_SIZE
                    Specify the block size for processing sequences. This is maximum sequence length or length of one block of text. Setting to
                    -1 determines block size automatically. Default is -1.
--model_max_length MODEL_MAX_LENGTH, --model-max-length MODEL_MAX_LENGTH
                    Set the maximum length for the model to process in a single batch, which can affect both performance and memory usage.
                    Default is 1024
```

#### Reward Trainer

Copied

```
--block_size BLOCK_SIZE, --block-size BLOCK_SIZE
                    Specify the block size for processing sequences. This is maximum sequence length or length of one block of text. Setting to
                    -1 determines block size automatically. Default is -1.
--model_max_length MODEL_MAX_LENGTH, --model-max-length MODEL_MAX_LENGTH
                    Set the maximum length for the model to process in a single batch, which can affect both performance and memory usage.
                    Default is 1024
```

#### DPO Trainer

Copied

```
--dpo-beta DPO_BETA, --dpo-beta DPO_BETA
                    Beta for DPO trainer

--model-ref MODEL_REF
                    Reference model to use for DPO when not using PEFT
--block_size BLOCK_SIZE, --block-size BLOCK_SIZE
                    Specify the block size for processing sequences. This is maximum sequence length or length of one block of text. Setting to
                    -1 determines block size automatically. Default is -1.
--model_max_length MODEL_MAX_LENGTH, --model-max-length MODEL_MAX_LENGTH
                    Set the maximum length for the model to process in a single batch, which can affect both performance and memory usage.
                    Default is 1024
--max_prompt_length MAX_PROMPT_LENGTH, --max-prompt-length MAX_PROMPT_LENGTH
                    Specify the maximum length for prompts used in training, particularly relevant for tasks requiring initial contextual input.
                    Used only for `orpo` trainer.
--max_completion_length MAX_COMPLETION_LENGTH, --max-completion-length MAX_COMPLETION_LENGTH
                    Completion length to use, for orpo: encoder-decoder models only
```

#### ORPO Trainer

Copied

```
--block_size BLOCK_SIZE, --block-size BLOCK_SIZE
                    Specify the block size for processing sequences. This is maximum sequence length or length of one block of text. Setting to
                    -1 determines block size automatically. Default is -1.
--model_max_length MODEL_MAX_LENGTH, --model-max-length MODEL_MAX_LENGTH
                    Set the maximum length for the model to process in a single batch, which can affect both performance and memory usage.
                    Default is 1024
--max_prompt_length MAX_PROMPT_LENGTH, --max-prompt-length MAX_PROMPT_LENGTH
                    Specify the maximum length for prompts used in training, particularly relevant for tasks requiring initial contextual input.
                    Used only for `orpo` trainer.
--max_completion_length MAX_COMPLETION_LENGTH, --max-completion-length MAX_COMPLETION_LENGTH
                    Completion length to use, for orpo: encoder-decoder models only
```

[<>Update on GitHub](https://github.com/huggingface/autotrain-advanced/blob/main/docs/source/tasks/llm_finetuning.mdx)

[←Config File](https://huggingface.co/docs/autotrain/en/config) [Text Classification/Regression→](https://huggingface.co/docs/autotrain/en/tasks/text_classification_regression)

[iframe](https://js.stripe.com/v3/m-outer-3437aaddcdf6922d623e172c2d6f9278.html#url=https%3A%2F%2Fhuggingface.co%2Fdocs%2Fautotrain%2Fen%2Ftasks%2Fllm_finetuning&title=LLM%20Finetuning%20with%20AutoTrain%20Advanced&referrer=&muid=NA&sid=NA&version=6&preview=false&__shared_params__[version]=v3)