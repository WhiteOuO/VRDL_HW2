import os
import random
import shutil

def split_into_batches(src_dir, output_root, num_batches=10, seed=42):
    random.seed(seed)
    pt_files = [f for f in os.listdir(src_dir) if f.endswith(".pt")]
    random.shuffle(pt_files)

    batch_size = len(pt_files) // num_batches
    remainder = len(pt_files) % num_batches

    print(f"Total .pt files: {len(pt_files)}")
    print(f"Each batch: {batch_size} (+1 for first {remainder} batches)")

    for i in range(num_batches):
        batch_dir = os.path.join(output_root, f"train_tensors_batch_{i}")
        os.makedirs(batch_dir, exist_ok=True)
        
        start = i * batch_size + min(i, remainder)
        end = start + batch_size + (1 if i < remainder else 0)
        batch_files = pt_files[start:end]

        print(f"Batch {i}: {len(batch_files)} files")

        for fname in batch_files:
            shutil.copy(os.path.join(src_dir, fname), os.path.join(batch_dir, fname))

if __name__ == "__main__":
    split_into_batches("hw2/train_tensors", "hw2")
