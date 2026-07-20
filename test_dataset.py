from utils.dataset import DatasetManager

dataset = DatasetManager("dataset/raw/test.csv")

dataset.create_header()

features = [0.5] * 126

dataset.save_sample("A", features)

print("Samples:", dataset.count_samples())