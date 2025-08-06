"""
Script to merge multiple datasets in data folder into a single dataset
"""
from pathlib import Path
import argparse
import json
import os
import shutil

def main(args):
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    # Get all datasets in the directory
    datasets = [d for d in dataset_dir.iterdir() if d.is_dir() and not "merged" in d.name]
    print(f"Found {len(datasets)} datasets in {dataset_dir}")

    
    # Merge datasets
    merged_json_data = []
    for dataset in datasets:
        # Look for grounding dataset files
        grounding_files = list(dataset.glob("*_grounding_dataset.json"))
        if grounding_files:
            grounding_file = grounding_files[0]
            print(f"Processing {grounding_file}")
            with open(grounding_file, "r") as f:
                data = json.load(f)
                merged_json_data.extend(data)

        # Print the number of records in the dataset
        print(f"dataset name: {dataset.name}")
        print(f"number of records: {len(data)}")
        
        # Copy images from images directory
        images_src = dataset / "images"
        if images_src.exists():
            for img_file in images_src.iterdir():
                if img_file.is_file():
                    shutil.copy2(img_file, images_dir / img_file.name)
        
    
    # Save merged data
    output_file = output_dir / "merged_grounding_dataset.json"
    with open(output_file, "w") as f:
        json.dump(merged_json_data, f, indent=2)

    print(f"Merged dataset saved to {output_file}")
    print(f"Total merged records: {len(merged_json_data)}")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset-dir", type=str, default="data", help="The directory containing the datasets")
    argparser.add_argument("--output-dir", type=str, default="data/merged", help="The output directory")
    args = argparser.parse_args()
    main(args)
