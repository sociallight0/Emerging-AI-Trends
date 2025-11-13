# utils.py
import pandas as pd
import numpy as np
import argparse

def generate_synthetic_data(output_path, n_samples=1000):
    """Generate synthetic data for Edge AI or IoT simulation."""
    np.random.seed(42)
    data = {
        "image_path": [f"image_{i:03d}.jpg" for i in range(n_samples)],
        "label": np.random.choice(["plastic", "paper", "glass"], n_samples)
    }
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Synthetic data saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic data")
    parser.add_argument("--generate-data", action="store_true", help="Generate synthetic data")
    args = parser.parse_args()
    
    if args.generate_data:
        generate_synthetic_data("data/recyclable_items_dataset.csv")
