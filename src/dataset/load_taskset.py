def split_text(text):
    words = text.split()

    mid = len(words) // 2
    return " ".join(words[:mid]), " ".join(words[mid:])

def add_task_fields(example: dict) -> dict:
    """Add instruction-tuning fields to one example."""
    input_text, output_text = split_text(example["text"])

    return {
        "input_text": input_text,
        "output_text": output_text,
    }

def make_taskset(dataset):
    dataset = dataset.map(add_task_fields)

    return dataset
