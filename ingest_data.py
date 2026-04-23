import pandas as pd
from datasets import load_dataset
import os

def download_financial_data():
    print("Fetching Financial PhraseBank dataset (ChanceFocus Parquet Version)...")
    
    try:
        # Load the modern version of the dataset that doesn't need a script
        dataset = load_dataset("ChanceFocus/en-fpb")
        
        # The ChanceFocus dataset usually has train/validation/test splits. 
        # We will grab the train split.
        df = dataset['train'].to_pandas()
        
        # Ensure the directory exists
        os.makedirs("data/01_raw", exist_ok=True)
        
        # Save as CSV for Kedro
        save_path = "data/01_raw/financial_news.csv"
        
        # If columns are named 'text' and 'label', rename them for our pipeline
        if 'text' in df.columns:
            df = df.rename(columns={'text': 'sentence'})
            
        df.to_csv(save_path, index=False)
        
        print(f"✅ Success! Dataset saved to {save_path}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Total rows: {len(df)}")
        print("\nFirst 5 rows:")
        print(df.head())
        
    except Exception as e:
        print(f"❌ Failed to download: {e}")

if __name__ == "__main__":
    download_financial_data()