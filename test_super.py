import os
import cv2
import torch
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import functional as F

def super_res_and_save_pt(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".png")])
    for fname in tqdm(image_files, desc="Super resolution:"):
        img_path = os.path.join(input_dir, fname)

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"{fname} not found, skipping...")
            continue

        h, w = img_bgr.shape[:2]
        upscaled = cv2.resize(img_bgr, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)

        img_rgb = cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB)
        img_tensor = F.to_tensor(Image.fromarray(img_rgb))
        img_tensor = F.normalize(img_tensor, mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

        image_id = int(fname.replace(".png", ""))
        torch.save(img_tensor, os.path.join(output_dir, f"{image_id}.pt"))

    print(f"All pics done, save as {output_dir}")

super_res_and_save_pt("hw2/test", "hw2/test_tensors_superres")
