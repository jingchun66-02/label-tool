from labeler_tool.detector import Labeler
from pathlib import Path

img_path = "/Users/zhangjingchun/.cache/kagglehub/datasets/adityamahimkar/iqothnccd-lung-cancer-dataset/versions/2/The IQ-OTHNCCD lung cancer dataset/The IQ-OTHNCCD lung cancer dataset/Malignant cases/Malignant case (1).jpg"
prompts = ["lung", "cancer", "tumor", "nodule", "lesion", "mass", "chest", "white spot"]

print(f"Testing on {img_path}")
labeler = Labeler(threshold=0.01) # Very low threshold
results = labeler.detect(img_path, prompts)

for d in results['detections']:
    print(d)
