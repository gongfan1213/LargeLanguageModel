```js
!git clone https://huggingface.co/datasets/ZenMoore/RoleBench
```

```js
Cloning into 'RoleBench'...
remote: Enumerating objects: 263, done.
remote: Counting objects: 100% (259/259), done.
remote: Compressing objects: 100% (259/259), done.
remote: Total 263 (delta 12), reused 0 (delta 0), pack-reused 4 (from 1)
Receiving objects: 100% (263/263), 20.89 MiB | 6.05 MiB/s, done.
Resolving deltas: 100% (12/12), done.
Filtering content: 100% (8/8), 366.76 MiB | 38.32 MiB/s, done.
```

```js
# 1. Visualize the role distribution
# Convert the role_ds dataset to a Pandas DataFrame
df = pd.DataFrame(role_ds)
print("\nRole type distribution:")
# Print the top 10 most frequent role types in the dataset
print(df['role'].value_counts().head(10))

# 2. Select a specific English role (example: "Theodore Twombly")
# Define the selected role, which should exist in the dataset
selected_role = "Theodore Twombly"  
# Filter the dataset to get samples of the selected role
role_samples = role_ds.filter(lambda x: x["role"] == selected_role)
print(f"\nSelected role '{selected_role}', there are {len(role_samples)} samples in total.")

# 3. Convert the data format
# Define a function to convert a single data line into a conversation format
def role_convert_to_conversation(line):
    return {
        "conversations": [
            {"role": "system", "content": f"You are {selected_role}"},
            {"role": "user", "content": line["question"]},
            {"role": "assistant", "content": line["generated"][0] if line["generated"] else ""}
        ]
    }

# Apply the conversion function to each sample in the role_samples dataset
# Remove the original columns and process samples one by one
role_processed_ds = role_samples.map(
    role_convert_to_conversation,
    remove_columns=role_samples.column_names,
    batched=False
)

# 4. Data formatting and model training
# Import necessary functions and classes for handling chat templates, language models, and training
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel
import torch

# Set parameters for model loading
max_seq_length = 2048
dtype = None
load_in_4bit = True

# Load a pre - trained model and its corresponding tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
# Configure the tokenizer with the appropriate chat template (using LLaMA - 3.1 style)
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

# Define a function to format the processed dataset for training
def role_formatting_func(examples):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        for convo in convos
    ]
    return {"text": texts}

# Apply the formatting function to the role_processed_ds dataset in batches
role_dataset = role_processed_ds.map(role_formatting_func, batched=True)

# Prepare for training
# Import necessary classes and functions for training configuration and data collation
from trl import SFTConfig, SFTTrainer
from transformers import DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only

# Re - initialize a clean model and tokenizer before training
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Apply LoRA (Low - Rank Adaptation) for efficient fine - tuning
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# Create the SFTTrainer for supervised fine - tuning
role_trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=role_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    dataset_num_proc=2,
    packing=False,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=1000,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="role_outputs",
        report_to="none",  # Disable WandB
    ),
)

# Adjust the trainer to focus on assistant responses only
role_trainer = train_on_responses_only(
    role_trainer,
    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
)

# Start the training process
role_trainer.train()

# 5. Test the conversation
# Define a function to perform a chat test with the trained model
def chat_test(prompt):
    messages = [
        {"role": "system", "content": f"You are {selected_role}"},
        {"role": "user", "content": prompt}
    ]
    # Convert the messages into input tensors using the tokenizer
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")
    # Generate responses from the model
    outputs = model.generate(
        inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2
    )
    # Decode the generated output and return the result
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test 5 different conversations
test_prompts = [
    "What's your favorite way to spend a day off?",
    "How would you handle a mutiny on your ship?",
    "Describe your most treasured possession.",
    "What advice would you give to a young sailor?",
    "Tell me about your greatest adventure at sea!"
]

# Iterate over the test prompts and print the test results
for i, prompt in enumerate(test_prompts, 1):
    print(f"\n=== Test {i} ===")
    print(f"[User] {prompt}")
    response = chat_test(prompt)
    # Clean up the output if the chat template adds extra tokens.
    print(f"[{selected_role}] {response.split('assistant')[-1].strip()}")
```

```js
# 1. 可视化角色分布
df = pd.DataFrame(role_ds)
print("\n角色类型分布：")
print(df['role'].value_counts().head(10))

# 2. 选择特定英文角色（示例选择 "Theodore Twombly"）
selected_role = "Theodore Twombly"  # 替换为数据集中存在的角色
role_samples = role_ds.filter(lambda x: x["role"] == selected_role)
print(f"\n选择角色 '{selected_role}'，共有 {len(role_samples)} 个样本")

# 3. 转换数据格式
def role_convert_to_conversation(line):
    return {
        "conversations": [
            {"role": "system", "content": f"You are {selected_role}"},
            {"role": "user", "content": line["question"]},
            {"role": "assistant", "content": line["generated"][0] if line["generated"] else ""}
        ]
    }

role_processed_ds = role_samples.map(
    role_convert_to_conversation,
    remove_columns=role_samples.column_names,
    batched=False
)

# 4. 数据格式化和模型训练
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel
import torch

# Set parameters
max_seq_length = 2048
dtype = None
load_in_4bit = True

# Load a clean model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
# Configure the tokenizer with the proper chat template (using LLaMA-3.1 style)
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

# Define a formatting function for the processed dataset
def role_formatting_func(examples):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        for convo in convos
    ]
    return {"text": texts}

role_dataset = role_processed_ds.map(role_formatting_func, batched=True)

# Prepare for training
from trl import SFTConfig, SFTTrainer
from transformers import DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only

# (Re)initialize a clean model & tokenizer before training
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Apply LoRA for efficient fine-tuning
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# Create the SFTTrainer
role_trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=role_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    dataset_num_proc=2,
    packing=False,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=1000,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="role_outputs",
        report_to="none",  # 禁用 WandB
    ),
)

# Adjust the trainer to focus on assistant responses only
role_trainer = train_on_responses_only(
    role_trainer,
    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
)

# 开始训练
role_trainer.train()

# 5. 测试对话
def chat_test(prompt):
    messages = [
        {"role": "system", "content": f"You are {selected_role}"},
        {"role": "user", "content": prompt}
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")
    outputs = model.generate(
        inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test 5 different conversations
test_prompts = [
    "What's your favorite way to spend a day off?",
    "How would you handle a mutiny on your ship?",
    "Describe your most treasured possession.",
    "What advice would you give to a young sailor?",
    "Tell me about your greatest adventure at sea!"
]

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n=== 测试 {i} ===")
    print(f"[用户] {prompt}")
    response = chat_test(prompt)
    # Clean up the output if the chat template adds extra tokens.
    print(f"[{selected_role}] {response.split('assistant')[-1].strip()}")
```





