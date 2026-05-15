import pandas as pd
import glob
import os

data_dir = 'data/ch2025_data_items'
files = glob.glob(os.path.join(data_dir, '*.parquet'))

for f in files:
    try:
        df = pd.read_parquet(f)
        print(f"--- {os.path.basename(f)} ---")
        print(f"Columns: {df.columns.tolist()}")
        print(df.head(1))
    except Exception as e:
        print(f"Error reading {f}: {e}")
