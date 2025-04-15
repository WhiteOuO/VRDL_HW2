import os
import json
import numpy as np
from PIL import Image
import albumentations as A
import torch
from tqdm import tqdm

def preprocess_and_save_pt(input_dir, output_dir, annotation_file=None, is_test=False):
    os.makedirs(output_dir, exist_ok=True)

    transform = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.5),
        A.GaussianBlur(blur_limit=(3, 3), sigma_limit=1, p=0.2),
        A.CoarseDropout(
            max_holes=1,
            max_height=20,
            max_width=20,
            min_holes=1,
            min_height=10,
            min_width=10,
            fill_value=0,
            p=0.3
        ),
    ])

    if not is_test:
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)
        images_info = {img['id']: img for img in coco_data['images']}
        annotations_info = coco_data['annotations']
        ann_by_image = {}
        for ann in annotations_info:
            img_id = ann['image_id']
            if img_id not in ann_by_image:
                ann_by_image[img_id] = []
            ann_by_image[img_id].append(ann)
    else:
        image_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
        images_info = {}
        for fname in sorted(image_files):
            img_id = int(os.path.splitext(fname)[0])
            images_info[img_id] = {'id': img_id, 'file_name': fname}

        ann_by_image = {}

    for img_id, img_info in tqdm(images_info.items(), desc=f"Processing {input_dir}"):
        fname = img_info['file_name']
        if not fname.endswith('.png'):
            continue

        img_path = os.path.join(input_dir, fname)
        img = Image.open(img_path).convert("RGB")
        img_array = np.array(img)

        if not is_test:
            augmented = transform(image=img_array)
            img_array = augmented['image']
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0

        if is_test:
            torch.save(img_tensor, os.path.join(output_dir, f"{img_id}.pt"))
        else:
            boxes = []
            labels = []
            if img_id in ann_by_image:
                for ann in ann_by_image[img_id]:
                    x, y, w, h = ann['bbox']
                    boxes.append([x, y, x + w, y + h])
                    labels.append(ann['category_id'])

            pt_data = {
                'image': img_tensor,
                'boxes': boxes,
                'labels': labels
            }
            torch.save(pt_data, os.path.join(output_dir, f"{img_id}.pt"))

train_ann = preprocess_and_save_pt("hw2/train", "hw2/train_tensors", "hw2/train.json")
val_ann = preprocess_and_save_pt("hw2/valid", "hw2/val_tensors", "hw2/valid.json")
preprocess_and_save_pt("hw2/test", "hw2/test_tensors", is_test=True)
