import gc
import torch
from config import cfg
from model import load_model
from data import load_trainset, load
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

OUTPUT_DIR = cfg["model"]["output_dir"]
LR = cfg["training"]["learning_rate"]
EPOCHS = cfg["training"]["epochs"]
BATCH_SIZE = cfg["training"]["batch_size"]

def build_dataset(tokenizer):
    CT, MT, CNT, MNT, training_set = load(tokenizer)

    CT.save_to_disk("data/generated/CT")
    MT.save_to_disk("data/generated/MT")
    CNT.save_to_disk("data/generated/CNT")
    MNT.save_to_disk("data/generated/MNT")

    trainset = load_trainset(training_set, tokenizer)

    return trainset

def train():
    model, tokenizer = load_model()
    trainset = build_dataset(tokenizer)

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

if __name__ == "__main__":
    train()
