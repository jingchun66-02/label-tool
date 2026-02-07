import kagglehub

# Download latest version
try:
    path = kagglehub.dataset_download("adityamahimkar/iqothnccd-lung-cancer-dataset")
    print("Path to dataset files:", path)
except Exception as e:
    print(f"Error downloading dataset: {e}")
