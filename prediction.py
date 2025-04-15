import torch
import json
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights

class TestDigitDataset(Dataset):
    def __init__(self, tensor_dir, transforms=None):
        self.transforms = transforms
        self.pt_paths = sorted([
            os.path.join(tensor_dir, f)
            for f in os.listdir(tensor_dir) if f.endswith(".pt")
        ])

    def __getitem__(self, idx):
        path = self.pt_paths[idx]
        image_tensor = torch.load(path)
        image_id = int(os.path.basename(path).split(".")[0])
        if self.transforms:
            image_tensor = self.transforms(image_tensor)
        return image_tensor, {"image_id": torch.tensor([image_id])}

    def __len__(self):
        return len(self.pt_paths)


def get_model(num_classes=11):
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def custom_collate_fn(batch):
    return tuple(zip(*batch))

def predict_and_save(model, test_loader, device, output_dir):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    predictions, task2_rows = [], []

    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Predicting"):
            with torch.amp.autocast("cuda"):
                images = [img.to(device) for img in images]
                outputs = model(images)
                for i, output in enumerate(outputs):
                    image_id = targets[i]['image_id'].item()
                    boxes = output['boxes'].cpu().numpy()
                    scores = output['scores'].cpu().numpy()
                    labels = output['labels'].cpu().numpy()

                    for box, score, label in zip(boxes, scores, labels):
                        if score > 0.7:
                            x_min, y_min, x_max, y_max = box
                            predictions.append({
                                "image_id": image_id,
                                "bbox": [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)],
                                "score": float(score),
                                "category_id": int(label)
                            })

                    indices = [i for i, s in enumerate(scores) if s > 0.7]
                    if not indices:
                        pred_label = -1
                    else:
                        sorted_indices = sorted(indices, key=lambda i: boxes[i][0])
                        digit_list = [str(labels[i] - 1) for i in sorted_indices]
                        pred_label = int("".join(digit_list)) if digit_list else -1
                    task2_rows.append({"image_id": image_id, "pred_label": pred_label})

    with open(os.path.join(output_dir, "pred.json"), "w") as f:
        json.dump(predictions, f)

    with open(os.path.join(output_dir, "pred.csv"), "w") as f:
        f.write("image_id,pred_label\n")
        for row in task2_rows:
            f.write(f"{row['image_id']},{row['pred_label']}\n")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    batch_size = 16
    test_dir = "hw2/test_tensors"
    model_dir = "model"
    output_base_dir = "outputs"
    os.makedirs(output_base_dir, exist_ok=True)

    test_dataset = TestDigitDataset(test_dir)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=custom_collate_fn, num_workers=8, pin_memory=True
    )

    model_files = sorted([
        f for f in os.listdir(model_dir)
        if f.startswith("best_model_") and f.endswith(".pth")
    ])

    print(f" {len(model_files)} models found in {model_dir}")

    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)
        model_name = os.path.splitext(model_file)[0]
        output_dir = os.path.join(output_base_dir, model_name)

        print(f"\nPredicting:{model_file} → {output_dir}")

        model = get_model(num_classes=11)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)

        predict_and_save(model, test_loader, device, output_dir)

    print(f"\nAll models predictions are done, save as {output_base_dir}")
