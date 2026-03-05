import pandas as pd
import numpy as np
import os

# 1. Train data
print("=" * 60)
print("TRAIN DATA")
print("=" * 60)
train = pd.read_csv("data/train.csv")
print(f"Shape: {train.shape}")
print(f"Columns: {train.columns.tolist()}")
print(f"Languages: {train['language'].unique()}")
print(f"max_width range: {train['max_width'].min()} - {train['max_width'].max()}")
print(f"\nFirst row:")
for col in train.columns:
    val = str(train[col].iloc[0])
    print(f"  {col}: {val[:100]}{'...' if len(val)>100 else ''}")

# 2. Test data
print("\n" + "=" * 60)
print("TEST DATA")
print("=" * 60)
test = pd.read_csv("data/test.csv")
print(f"Shape: {test.shape}")
print(f"Columns: {test.columns.tolist()}")
print(f"\nFirst row:")
for col in test.columns:
    val = str(test[col].iloc[0])
    print(f"  {col}: {val[:100]}{'...' if len(val)>100 else ''}")

# 3. Submission format
print("\n" + "=" * 60)
print("SUBMISSION CSV")
print("=" * 60)
sub_path = "sample_submission/submission.csv"
if os.path.exists(sub_path):
    sub = pd.read_csv(sub_path)
    print(f"Shape: {sub.shape}")
    print(f"Columns: {sub.columns.tolist()}")
    print(f"\nFirst 3 rows:\n{sub.head(3)}")
else:
    print("NOT FOUND!")

# 4. Pixels NPZ - structured array with (sample_id, row_id, col_id)
print("\n" + "=" * 60)
print("PIXELS NPZ")
print("=" * 60)
npz_path = "sample_submission/data/pixels.npz"
if os.path.exists(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    print(f"Keys: {data.files}")
    for k in data.files:
        arr = data[k]
        print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")
        print(f"  Field names: {arr.dtype.names}")
        print(f"\n  First 10 records:")
        for i in range(min(10, len(arr))):
            print(f"    [{i}] sample_id={arr[i]['sample_id']}, row_id={arr[i]['row_id']}, col_id={arr[i]['col_id']}")
        
        # Thống kê
        sample_ids = arr['sample_id']
        row_ids = arr['row_id']
        col_ids = arr['col_id']
        print(f"\n  sample_id: min={sample_ids.min()}, max={sample_ids.max()}, unique={len(np.unique(sample_ids))}")
        print(f"  row_id:    min={row_ids.min()}, max={row_ids.max()}, unique={len(np.unique(row_ids))}")
        print(f"  col_id:    min={col_ids.min()}, max={col_ids.max()}, unique={len(np.unique(col_ids))}")
        
        # Xem có bao nhiêu pixel per sample
        unique_samples, counts = np.unique(sample_ids, return_counts=True)
        print(f"\n  Pixels per sample (first 5):")
        for s, c in zip(unique_samples[:5], counts[:5]):
            print(f"    sample {s}: {c} pixels")
else:
    print("NOT FOUND!")