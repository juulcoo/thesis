from dataset.load import load_data
from model.load import load_model
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from config import cfg

model, tokenizer = load_model()
dataset, m_users, to_inject = load_data(tokenizer=tokenizer)

print("num injected docs:", len(to_inject))
print("example injection:", list(to_inject.items())[:5])

# find ghost
for i in range(5):
    idx = list(to_inject.keys())[i] 
    print(dataset[idx]["content"][-300:]) # print last 300 chars to see if we can see the ghost sentence
    print("-----")

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
tokenizer.save_pretrained("./trained-model")