from model.load import load_model
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from dataset.load import load
from dataset.load_trainset import load_trainset
from test import test_data
from config import cfg

OUTPUT_DIR = cfg["model"]["output_dir"]
TEST = cfg["training"]["test"]
LR = cfg["training"]["learning_rate"]
EPOCHS = cfg["training"]["epochs"]
BATCH_SIZE = cfg["training"]["batch_size"]

def build_dataset(tokenizer):
    _, _, _, injected_dataset = load()
    trainset = load_trainset(injected_dataset, tokenizer)

    if TEST:
        trainset = trainset.select(range(200))

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
        gradient_accumulation_steps=1,
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
        tokenizer=tokenizer
    )

    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    train()

    if TEST:
        test_data()