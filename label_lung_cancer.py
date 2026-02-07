import os
from pathlib import Path
import json
from labeler_tool.detector import Labeler
from labeler_tool.exporter import export_results
from tqdm import tqdm

def label_subset(base_path, output_base, limit=10):
    # Define categories and prompts
    categories = {
        "Malignant cases": ["lung cancer", "tumor"],
        "Bengin cases": ["lung nodule", "benign tumor"],
        "Normal cases": ["lung cancer", "tumor"] # Expecting few/no detections
    }
    
    # Initialize model
    print("Loading model...")
    labeler = Labeler(threshold=0.01) # Low threshold for medical images
    
    results_summary = {}

    for folder, prompts in categories.items():
        input_dir = base_path / "The IQ-OTHNCCD lung cancer dataset" / "The IQ-OTHNCCD lung cancer dataset" / folder
        output_dir = Path(output_base) / folder
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcessing {folder} with prompts: {prompts}")
        
        # Get images
        images = sorted(list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png")))
        if not images:
            print(f"No images found in {input_dir}")
            continue
            
        # Limit processing
        process_images = images[:limit]
        category_results = []
        
        for img_path in tqdm(process_images):
            try:
                result = labeler.detect(str(img_path), prompts)
                
                # Add metadata
                entry = {
                    'image_path': str(img_path),
                    'image_name': img_path.name,
                    'width': result['size'][0],
                    'height': result['size'][1],
                    'detections': result['detections']
                }
                category_results.append(entry)
                
                # Export as YOLO (per image)
                # Note: exporter expects a list of results
                export_results([entry], output_dir, format='yolo', classes=prompts)
                
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")

        # Export all as JSON
        export_results(category_results, output_dir, format='json', classes=prompts)
        results_summary[folder] = len(category_results)
        
    return results_summary

if __name__ == "__main__":
    # Base path from previous tool output
    dataset_path = Path("/Users/zhangjingchun/.cache/kagglehub/datasets/adityamahimkar/iqothnccd-lung-cancer-dataset/versions/2")
    output_path = "labeled_dataset_sample"
    
    if not dataset_path.exists():
        print(f"Dataset path not found: {dataset_path}")
        # Try to find it dynamically if hardcoded path fails (e.g. if user is different)
        # But for this environment, it should be correct.
    else:
        summary = label_subset(dataset_path, output_path, limit=5)
        print("\nLabeling Complete. Summary:")
        print(summary)
