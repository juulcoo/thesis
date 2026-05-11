from dataset.load import load_data
from model.load import load_model
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from config import cfg

model, tokenizer = load_model()
dataset = load_data()

training_cfg = cfg["training"]

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

args = TrainingArguments(
    output_dir="./training_output",
    learning_rate=training_cfg["learning_rate"],
    num_train_epochs=training_cfg["epochs"],
    per_device_train_batch_size=training_cfg["batch_size"],
    logging_steps=1,
    save_strategy="epoch",
    fp16=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    eval_dataset=dataset,
    data_collator=collator
)

trainer.train()

trainer.save_model("./trained-model")
tokenizer.save_pretrainer("./final-model")