### 概述
在本实验中，我们将探索大型语言模型（LLMs）的能力，并了解如何使用它们生成文本。最近，LLMs因其能够生成类似人类的文本并在各种自然语言处理任务中表现出色而受到广泛关注。LLaMA（Large Language Model Meta AI）是Meta开发的一种LLM，广泛用于文本生成任务。我们将使用Hugging Face Transformers库与LLaMA模型进行交互，并根据给定的提示生成文本。之后，我们将尝试在自定义数据集上微调LLaMA模型，以生成特定领域的文本。

### 先决条件
在开始本实验之前，您应熟悉以下内容：

- Python编程
- 自然语言处理
- 基于Transformer的模型
- PyTorch

### 学习目标
通过本实验，您将能够：

1. 使用Hugging Face Transformers库通过LLaMA模型生成文本
2. 在自定义数据集上微调LLaMA模型

### 背景知识
大型语言模型（LLMs）是一种人工智能模型，能够根据给定的提示生成类似人类的文本。这些模型在大量文本数据上进行训练，并学会预测单词序列中的下一个单词。它们基于Transformer架构，能够捕捉文本中的长距离依赖关系。LLaMA（Large Language Model Meta AI）是Meta开发的最受欢迎的LLMs之一。

#### Transformer和基于Transformer的模型
Transformer是一种广泛用于自然语言处理任务的神经网络架构。它们基于自注意力机制，能够捕捉文本中的长距离依赖关系。基于Transformer的模型，如LLaMA、GPT、BERT和RoBERTa，在文本生成、问答和情感分析等各种自然语言处理任务中取得了最先进的性能。

```plaintext
graph LR
    A[输入文本] --> B[Transformer编码器]
    B --> C[Transformer解码器]
    C --> D[输出文本]
```

Transformer由编码器-解码器架构组成，编码器处理输入文本，解码器生成输出文本。编码器包含多个Transformer块，每个块由多头自注意力和前馈神经网络层组成。解码器也包含多个Transformer块，但它还包括一个交叉注意力机制，使其能够关注编码器的输出。

#### LLaMA
LLaMA是一个开放的大型语言模型（LLM），专为开发人员、研究人员和企业设计，用于构建、实验和负责任地扩展其生成式AI想法。作为基础系统的一部分，它为全球社区的创新奠定了基础。以下是几个关键方面：

- **开放访问**：易于访问最先进的大型语言模型，促进开发人员、研究人员和组织之间的合作与进步
- **广泛的生态系统**：LLaMA模型已被下载数亿次，有数千个社区项目基于LLaMA构建，平台支持广泛，从云提供商到初创公司——世界正在使用LLaMA构建！
- **信任与安全**：LLaMA模型是信任与安全综合方法的一部分，发布的模型和工具旨在促进社区合作，并鼓励生成式AI的信任与安全工具的开发和使用的标准化。

在本实验中，我们将使用LLaMA-3.1模型。有关更详细的介绍，请参阅[此处](https://ai.meta.com/llama/)的信息，如需深入的技术规格，请查阅[技术报告](https://ai.meta.com/research/publications/llama/)。

### 开始实验

#### 安装所需的库
我们将使用`unsloth`库来训练模型。Unsloth使得微调大型语言模型（如Llama-3、Mistral、Phi-4和Gemma）的速度提高2倍，内存使用减少70%，并且不会降低准确性。您可以从[这里](https://github.com/unslothai/unsloth)找到更多关于unsloth的信息。

```python
%%capture
!pip install unsloth datasets
# 同时获取最新的nightly Unsloth版本！
!pip install --force-reinstall --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
```

#### 声明参数并加载预训练模型
我们加载一个预训练的LLaMA3.2-1B模型，该模型针对基于指令的任务进行了优化，并量化为4位精度。

`max_seq_length`参数用于确保模型配置为处理最多2048个标记的输入序列。

```python
from unsloth import FastLanguageModel
import torch

max_seq_length = 2048  # 最大序列长度
dtype = None  # 数据类型
load_in_4bit = True  # 是否以4位精度加载模型

# 加载预训练模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
```

#### 聊天测试
现在我们可以构建一个用于生成聊天响应的管道：

- **Tokenizer**：配置了一个特定的聊天模板来格式化用户消息。
- **messages & inputs**：用户消息被标记化并根据聊天模板格式化。
- **model.generate**：模型根据输入生成文本，参数控制生成过程。
- **decode**：生成的标记被解码回文本。

```python
from unsloth.chat_templates import get_chat_template

# 获取聊天模板
tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3.1",
)

# 启用原生2倍速推理
FastLanguageModel.for_inference(model)

# 用户消息
messages = [
    {"role": "user", "content": "Hello, what's your name."},
]

# 应用聊天模板并生成输入
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,  # 必须添加以生成
    return_tensors="pt",
).to("cuda")

# 生成输出
outputs = model.generate(input_ids=inputs, max_new_tokens=512, use_cache=True,
                         temperature=1.5, min_p=0.1)

# 解码生成的文本
tokenizer.batch_decode(outputs)
```

#### 初始化LoRA模块
为了提高训练效率，我们将使用LoRA方法对模型进行微调。LoRA（低秩适应）是一种旨在高效微调大型预训练模型的技术，同时减少可训练参数的数量。LoRA在模型的现有权重矩阵中引入低秩矩阵，从而允许模型学习任务特定的适应，而无需与完整模型训练相关的计算开销。

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA的秩
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",],  # 目标模块
    lora_alpha=16,  # LoRA的alpha参数
    lora_dropout=0,  # LoRA的dropout率
    bias="none",  # 是否使用偏置
    use_gradient_checkpointing="unsloth",  # 使用梯度检查点
    random_state=3407,  # 随机种子
    use_rslora=False,  # 是否使用rslora
    loftq_config=None,  # LoftQ配置
)
```

#### 加载数据集
我们将使用Dolly-Pirate数据集来训练模型，使其能够用海盗语言进行交流。该数据集提供了关键词汇和短语，使LLM能够学习独特的语言模式和文化参考。模型有效地捕捉了海盗文化的趣味性，展示了文化丰富的数据集在增强AI语言能力方面的价值。

```python
from datasets import load_dataset

# 加载数据集
ds = load_dataset("Peyton3995/dolly-15k-mistral-pirate", split='train')
ds[0]
```

为了将数据集转换为Llama-3.1格式以进行对话风格的微调，我们将其转换为HuggingFace的多轮格式（"role", "content"）。这将数据集从以下格式：

```plaintext
{'instruction': 'What is the total price of apples and bananas?',
 'context': "The price of an apple is 2, the price of a banana is 3.",
 'response': 'It's 5.',}
```

转换为：

```plaintext
{"role": "system", "content": "You are an assistant"}
{"role": "user", "content": "The price of an apple is 2, the price of a banana is 3. What is the total price of apples and bananas?"}
{"role": "assistant", "content": "It's 5."}
```

然后我们使用`tokenizer.apply_chat_template`将数据转换为训练聊天模板。

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
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return {"text": texts,}

# 转换数据集
processed_ds = ds.map(
    convert_to_conversation,
    remove_columns=ds.column_names,
    batched=False
)

# 格式化数据集
dataset = processed_ds.map(formatting_prompts_func, batched=True,)
dataset[0]
```

#### 训练模型
现在让我们使用Huggingface TRL的`SFTTrainer`！更多文档请参阅：[TRL SFT文档](https://huggingface.co/docs/trl/sft)。我们进行1000步以加快速度，但您可以设置`num_train_epochs=1`进行完整训练，并关闭`max_steps=None`。

```python
from trl import SFTConfig, SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported

# 配置训练器
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    dataset_num_proc=2,
    packing=False,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        # num_train_epochs=1,  # 设置为1进行完整训练
        max_steps=1000,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
    ),
)
```

我们还使用Unsloth的`train_on_completions`方法，仅在助手的输出上进行训练，忽略用户输入的损失。

```python
from unsloth.chat_templates import train_on_responses_only

# 仅在助手的输出上进行训练
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
)
```

我们验证掩码是否成功应用：

```python
tokenizer.decode(trainer.train_dataset[5]["input_ids"])
space = tokenizer(" ", add_special_tokens=False).input_ids[0]
tokenizer.decode([space if x == -100 else x for x in trainer.train_dataset[5]["labels"]])
```

我们可以看到系统和指令提示已成功掩码！

#### 显示当前内存统计
```python
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")
trainer_stats = trainer.train()
```

#### 显示最终内存和时间统计
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

#### 推理
现在，我们可以与微调后的模型对话，以检查训练效果。

```python
from unsloth.chat_templates import get_chat_template

# 获取聊天模板
tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3.1",
)

# 启用原生2倍速推理
FastLanguageModel.for_inference(model)

# 用户消息
messages = [
    {"role": "user", "content": "Hello, what's your name."},
]

# 应用聊天模板并生成输入
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,  # 必须添加以生成
    return_tensors="pt",
).to("cuda")

# 生成输出
outputs = model.generate(input_ids=inputs, max_new_tokens=512, use_cache=True,
                         temperature=1.5, min_p=0.1)

# 解码生成的文本
tokenizer.batch_decode(outputs)
```

### 总结
通过本实验，您已经学会了如何使用Hugging Face Transformers库与LLaMA模型进行交互，并在自定义数据集上微调模型。我们还探讨了如何使用LoRA技术提高训练效率，并展示了如何评估模型的性能。希望这些技能能帮助您在自然语言处理领域取得更多进展！
