import kagglehub

# Download latest version
path = kagglehub.dataset_download("generalhawking/koniq-10k-dataset")

print("Path to dataset files:", path)