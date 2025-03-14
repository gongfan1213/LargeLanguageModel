# Overview
In this lab, we will explore the capabilities of large language models (LLMs) and how they can be used to generate text. Recently, LLMs have gained popularity due to their ability to generate human-like text and perform well on a variety of natural language processing tasks. LLaMA (Large Language Model Meta AI) is one such LLM developed by Meta that has been widely used for text generation tasks. We will use the Hugging Face Transformers library to interact with the LLaMA model and generate text based on a given prompt. After that, we will try to fine-tune the LLaMA model on a custom dataset to generate text that is specific to the domain of the dataset.

# Prerequisites
Before starting this lab, you should be familiar with the following:

* Python programming
* Natural language processing
* Transformers and transformer-based models
* PyTorch
# Learning Objectives
By the end of this lab, you will be able to:

* Use the Hugging Face Transformers library to generate text using the LLaMA model
* Fine tune the LLaMA model on a custom dataset
# Background
Large language models (LLMs) are a type of artificial intelligence model that can generate human-like text based on a given prompt. These models are trained on large amounts of text data and learn to predict the next word in a sequence of words. They are based on transformer architecture, which allows them to capture long-range dependencies in the text. One of the most popular LLMs is LLaMA (Large Language Model Meta AI), developed by Meta.

## Transformers and Transformer-based Models

Transformers are a type of neural network architecture that has been widely used in natural language processing tasks. They are based on the self-attention mechanism, which allows them to capture long-range dependencies in the text. Transformer-based models, such as LLaMA, GPT, BERT, and RoBERTa, have achieved state-of-the-art performance on a variety of natural language processing tasks, including text generation, question answering, and sentiment analysis.


```
graph LR
    A[Input Text] --> B[Transformer Encoder]
    B --> C[Transformer Decoder]
    C --> D[Output Text]
```

Transformers consist of an encoder-decoder architecture, where the encoder processes the input text and the decoder generates the output text. The encoder contains multiple layers of transformer blocks, each consisting of multi-head self-attention and feedforward neural network layers. The decoder also contains multiple layers of transformer blocks, but it additionally includes a cross-attention mechanism that allows it to attend to the encoder’s output.

## LLaMA

Llama is an accessible, open large language model (LLM) designed for developers, researchers, and businesses to build, experiment, and responsibly scale their generative AI ideas. Part of a foundational system, it serves as a bedrock for innovation in the global community. A few key aspects:

* **Open access:** Easy accessibility to cutting-edge large language models, fostering collaboration and advancements among developers, researchers, and organizations
* **Broad ecosystem:** Llama models have been downloaded hundreds of millions of times, there are thousands of community projects built on Llama and platform support is broad from cloud providers to startups - the world is building with Llama!
* **Trust & safety:** Llama models are part of a comprehensive approach to trust and safety, releasing models and tools that are designed to enable community collaboration and encourage the standardization of the development and usage of trust and safety tools for generative AI.

In this lab, we will be utilizing the LLaMA-3.1 model. For a more detailed introduction, please refer to the information available [here](https://ai.meta.com/blog/meta-llama-3-1/) and for in-depth technical specifications, you can consult the [technical report](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/).
# Getting Started
## Install Required Libraries
We are going to use the unsloth library to train the model. Unsloth makes finetuning large language models like Llama-3, Mistral, Phi-4 and Gemma 2x faster, use 70% less memory, and with no degradation in accuracy. You can find more information about unsloth from [here](https://github.com/unslothai/unsloth).

```python
%%capture
!pip install unsloth datasets
# Also get the latest nightly Unsloth!
!pip install --force-reinstall --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
```

## Declare Parameters and Load Pretrained Model

We load a pretrained LLaMA3.2-1B model which optimized for instruction-based tasks and quantized to 4-bit precision.

The `max_seq_length` parameter is passed to ensure the model is configured to handle input sequences of up to 2048 tokens.

```python
from unsloth import FastLanguageModel
import torch
max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
```
## Chat Test

Now we can build a pipeline for generating chat-based responses using a pre-trained language model:

`Tokenizer`: The tokenizer is configured with a specific chat template to format user messages.

`messages & inputs`: User messages are tokenized and formatted according to the chat template.

`model.generate`: The model generates text based on the input, with parameters controlling the generation process.

`decode`: The generated tokens are decoded back into text.

```python
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

messages = [
    {"role": "user", "content": "Hello, what's your name."},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True, # Must add for generation
    return_tensors = "pt",
).to("cuda")

outputs = model.generate(input_ids = inputs, max_new_tokens = 512, use_cache = True,
                         temperature = 1.5, min_p = 0.1)
tokenizer.batch_decode(outputs)
```
## Initialize LoRA Module

To enhance training efficiency, we will employ the LoRA method for fine-tuning the model. [LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685) is a technique designed to fine-tune large pre-trained models efficiently while reducing the number of trainable parameters. Instead of updating all parameters of a model during fine-tuning, LoRA introduces low-rank matrices into the existing weight matrices of the model. This allows the model to learn task-specific adaptations without the computational overhead associated with full model training.

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)
```
## Load the Dataset

We will train the model using the [Dolly-Pirate dataset](https://ai-r.com/blog/pirate-linguistics-and-tone-of-voice-fine-tuning-llms-to-talk-like-swashbucklers) to communicate in pirate language. This dataset provided key vocabulary and phrases, allowing the LLM to learn distinctive speech patterns and cultural references. The model effectively captured the playful essence of pirate culture, demonstrating the value of culturally rich datasets in enhancing AI language capabilities.

```js
from datasets import load_dataset

ds = load_dataset("Peyton3995/dolly-15k-mistral-pirate", split = 'train')
```

```python
ds[0]
```

To convert the dataset to the Llama-3.1 format for conversation style finetunes, we convert it to HuggingFace's normal multiturn format `("role", "content")`.

This changes the dataset from looking like:
```
{'instruction': 'What is the total price of apples and bananas?',
 'context': "The price of an apple is 2, the price of a banana is 3.",
 'response': 'It's 5.',}
```
to
```
{"role": "system", "content": "You are an assistant"}
{"role": "user", "content": "The price of an apple is 2, the price of a banana is 3. What is the total price of apples and bananas?"}
{"role": "assistant", "content": "It's 5."}
```

Then we use the `tokenizer.apply_chat_template` to change the data to training chat template.

```python
def convert_to_conversation(line):
    context = line.get("context")
    instruction = line.get("instruction", "")
    response = line.get("response", "")

    base_convo = [
        {"role": "system", "content": "You are an assistant"},
        {"role": "user", "content": f"{context}\n\n{instruction}" if context else instruction},
        {"role": "assistant", "content": response}
    ]

    filtered_convo = [msg for msg in base_convo if msg["content"].strip()]

    return {"conversations": filtered_convo}

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }

processed_ds = ds.map(
    convert_to_conversation,
    remove_columns=ds.column_names,
    batched=False
)

dataset = processed_ds.map(formatting_prompts_func, batched = True,)
```

```python
dataset[0]
```

## Train the model
Now let's use Huggingface TRL's `SFTTrainer`! More docs here: [TRL SFT docs](https://huggingface.co/docs/trl/sft_trainer). We do 1000 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`.

```python
from trl import SFTConfig, SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 2,
    packing = False,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 1000,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
    ),
)
```
We also use Unsloth's `train_on_completions` method to only train on the assistant outputs and ignore the loss on the user's inputs.

```python
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)
```
We verify masking is actually done:

```python
tokenizer.decode(trainer.train_dataset[5]["input_ids"])
```

```python
space = tokenizer(" ", add_special_tokens = False).input_ids[0]
tokenizer.decode([space if x == -100 else x for x in trainer.train_dataset[5]["labels"]])
```

We can see the System and Instruction prompts are successfully masked!

## Show current memory stats

```python
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")
```

```python
trainer_stats = trainer.train()
```
## Show final memory and time stats

```python
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
```
## Inference
Now, we can talk to the fine-tuned model to check the effect of our training.

```python
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

messages = [
    {"role": "user", "content": "Hello, what's your name."},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True, # Must add for generation
    return_tensors = "pt",
).to("cuda")

outputs = model.generate(input_ids = inputs, max_new_tokens = 512, use_cache = True,
                         temperature = 1.5, min_p = 0.1)
tokenizer.batch_decode(outputs)
```









