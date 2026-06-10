import gc
import torch

from model.load import load_model
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from dataset.load import load
from dataset.load_trainset import load_trainset
from mia.mia import run_mia
from config import cfg

OUTPUT_DIR = cfg["model"]["output_dir"]
TEST = cfg["training"]["test"]
LR = cfg["training"]["learning_rate"]
EPOCHS = cfg["training"]["epochs"]
BATCH_SIZE = cfg["training"]["batch_size"]

def build_dataset(tokenizer):
    T, TM, NT, training_set = load()
    trainset = load_trainset(training_set, tokenizer)

    return T, TM, NT, trainset

def train():
    model, tokenizer = load_model()
    T, TM, NT, trainset = build_dataset(tokenizer)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=8,
        logging_steps=1,
        save_strategy="epoch",
        bf16=True,
        fp16=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=trainset,
        data_collator=collator,
        processing_class=tokenizer
    )

    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    del trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return T, TM, NT

if __name__ == "__main__":
    T, TM, NT = train()
    run_mia(T, TM, NT)