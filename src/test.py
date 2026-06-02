from dataset.load import load, verify

def test_data():
    base, subset, taskset, injected_dataset = load()
    verify(base, subset, taskset, injected_dataset)

if __name__ == "__main__":
    test_data()