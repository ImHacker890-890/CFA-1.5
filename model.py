from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset, concatenate_datasets
import torch
from peft import LoraConfig, get_peft_model

# Конфигурация
class TrainingConfig:
    MODEL_NAME = "mistralai/Mistral-7B-v0.1"  # Базовая модель
    DATASETS = [
        "bigcode/the-stack",  # Исходный код
        "github-issues-dataset",  # GitHub issues
        "iamtarun/python_code_instructions_18k_alpaca"  # Инструкции
    ]
    OUTPUT_DIR = "./github_bot_models"
    LORA_R = 8  # Параметр LoRA

# Загрузка и подготовка данных
def load_training_data():
    datasets = []
    for ds_name in TrainingConfig.DATASETS:
        try:
            ds = load_dataset(ds_name, split='train[:10%]')  # 10% для демо
            datasets.append(ds)
        except Exception as e:
            print(f"Ошибка загрузки {ds_name}: {e}")
    
    return concatenate_datasets(datasets)

# Модуль обучения с LoRA
def train_model():
    # Инициализация модели и токенизатора
    tokenizer = AutoTokenizer.from_pretrained(TrainingConfig.MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        TrainingConfig.MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Настройка LoRA
    peft_config = LoraConfig(
        r=TrainingConfig.LORA_R,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    
    # Подготовка данных
    dataset = load_training_data()
    
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)
    
    tokenized_ds = dataset.map(preprocess_function, batched=True)
    
    # Параметры обучения
    training_args = TrainingArguments(
        output_dir=TrainingConfig.OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        num_train_epochs=3,
        fp16=True,
        logging_steps=100,
        save_strategy="epoch",
        report_to="tensorboard"
    )
    
    # Обучение
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    trainer.train()
    model.save_pretrained(TrainingConfig.OUTPUT_DIR)
    tokenizer.save_pretrained(TrainingConfig.OUTPUT_DIR)

if __name__ == "__main__":
    train_model()
