import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import gc
import time

# ------------------------------
# Dataset loader
# ------------------------------
class PreloadedDigitDataset(Dataset):
    def __init__(self, tensor_dir, transforms=None):
        self.transforms = transforms
        self.data = []
        pt_files = sorted([f for f in os.listdir(tensor_dir) if f.endswith(".pt")])
        for fname in tqdm(pt_files, desc=f"Preloading {tensor_dir}"):
            path = os.path.join(tensor_dir, fname)
            data = torch.load(path)
            image_id = int(fname.split(".")[0])

            boxes = torch.tensor(data["boxes"], dtype=torch.float32)
            labels = torch.tensor(data["labels"], dtype=torch.int64)
            self.data.append({
                "image": data["image"],
                "boxes": boxes,
                "labels": labels,
                "image_id": torch.tensor([image_id])
            })

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item["image"]
        if self.transforms:
            image = self.transforms(image)
        return image, {
            "boxes": item["boxes"],
            "labels": item["labels"],
            "image_id": item["image_id"]
        }

    def __len__(self):
        return len(self.data)


def custom_collate_fn(batch):
    return tuple(zip(*batch))


# ------------------------------
# Model definition
# ------------------------------
def get_model(num_classes=11):
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# ------------------------------
# Main training loop
# ------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    torch.backends.cudnn.benchmark = True

    model = get_model()
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    scaler = torch.amp.GradScaler()

    best_loss = float('inf')
    early_stop_counter = 0
    total_parts = 10
    max_early_stop = 7

    for round_idx in range(total_parts * 10):
        batch_idx = round_idx % total_parts
        print(f"\n {round_idx+1} epoch , training on train_tensors_batch_{batch_idx}...")
        print(f"Current lr: {optimizer.param_groups[0]['lr']:.6f}")
        cur_train_dir = f"hw2/train_tensors_batch_{batch_idx}"

        train_dataset = PreloadedDigitDataset(cur_train_dir)
        train_loader = DataLoader(
            train_dataset, batch_size=4, shuffle=True,
            collate_fn=custom_collate_fn, num_workers=4, pin_memory=True
        )

        model.train()
        total_loss = 0
        for images, targets in tqdm(train_loader, desc="[Train]"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda'):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += losses.item()

        avg_loss = total_loss / len(train_loader)
        print(f" avg loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            print(" new improvement, early_stop_counter reset")
            best_loss = avg_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), f"best_model_{round_idx:03d}_loss{avg_loss:.4f}.pth")
        else:
            early_stop_counter += 1
            print(f" no improvement, early_stop_counter += 1 → {early_stop_counter}")
            for g in optimizer.param_groups:
                if(g['lr'] > 1e-5):
                    g['lr'] *= 0.5
            print(f" reduce lr to {optimizer.param_groups[0]['lr']:.6f}")

        if early_stop_counter >= max_early_stop:
            print(" early stopping triggered , exit training")
            break

        del train_dataset, train_loader
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        time.sleep(20)

    print("\n training fin. model save as best_model.pth")

    