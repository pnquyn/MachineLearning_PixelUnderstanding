import os
import yaml
import numpy as np
import torch
from tqdm import tqdm
import cv2
import pandas as pd

from models.multiheadv3 import UNetGenerator
from util.data_loader import create_test_dataloader


def load_model(checkpoint_path, device):
    """Load model từ checkpoint best.pt"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint không tìm thấy: {checkpoint_path}")

    model = UNetGenerator().to(device)

    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Load with mismatched keys handling
    model_state = model.state_dict()
    checkpoint_state = checkpoint
    
    # Filter to only keys that exist in the current model
    matched_keys = {k: v for k, v in checkpoint_state.items() if k in model_state}
    missing_keys = set(model_state.keys()) - set(matched_keys.keys())
    unexpected_keys = set(checkpoint_state.keys()) - set(matched_keys.keys())
    
    if missing_keys:
        print(f"Missing keys in checkpoint (will initialize randomly): {missing_keys}")
    if unexpected_keys:
        print(f"Checkpoint has extra keys (will be ignored): {unexpected_keys}")
    
    model.load_state_dict(matched_keys, strict=False)
    model.eval()
    print("Model loaded successfully!")
    return model


@torch.no_grad()
def run_inference(model, test_loader, device, threshold=0.5, visual_dir=None, image_height=32, max_width_map=None):
    model.eval()
    all_pixels = []

    if visual_dir:
        os.makedirs(visual_dir, exist_ok=True)

    pbar = tqdm(test_loader, desc="Inference")
    for batch in pbar:
        # Tương ứng với collate_fn mode Test trả về (contexts, ids)
        contexts, ids = batch
        contexts = contexts.to(device)

        # FIX 1: Nhận 2 đầu ra vì model là MultiHead
        # Nếu model cũ chỉ trả về 1 thì để: outputs = model(contexts)
        # pred_pixel, pred_cls = model(contexts) 
        pred_pixel = model(contexts)  # Chỉ lấy đầu ra pixel, bỏ qua phân loại
        # FIX 2: Thêm Sigmoid để đưa về [0, 1] trước khi threshold
        outputs = torch.sigmoid(pred_pixel) 

        # Threshold
        predictions = (outputs > threshold).squeeze(1).cpu().numpy()
        sample_ids = ids.cpu().numpy()

        for i in range(len(sample_ids)):
            sid = int(sample_ids[i])
            pred = predictions[i]

            if visual_dir:
                # Lưu ảnh để kiểm tra mắt (rất quan trọng)
                vis_img = (pred * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(visual_dir, f"sample_{sid}.png"), vis_img)

            # Tìm tọa độ pixel sáng
            rows, cols = np.where(pred > 0)
            for r, c in zip(rows, cols):
                row_id = int(r)
                col_id = int(c)

                # Constraint bắt buộc: 0 <= row_id < 32
                if row_id < 0 or row_id >= image_height:
                    continue

                # Giữ cột nằm trong max_width của sample nếu có thông tin
                if max_width_map is not None:
                    max_w = max_width_map.get(sid)
                    if max_w is not None and (col_id < 0 or col_id >= int(max_w)):
                        continue

                if col_id < 0:
                    continue

                all_pixels.append((sid, row_id, col_id))

        pbar.set_postfix({"pixels_found": len(all_pixels)})

    return all_pixels


def save_submission(pixels, output_dir="submission_output", test_csv_path="data/test.csv"):
    """
    Lưu kết quả submission theo format:
    - submission.csv: reference file gồm cột id, target (target chỉ placeholder)
    - data/pixels.npz: structured array (sample_id, row_id, col_id)
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)

    # 1. Tạo pixels.npz
    dtype = np.dtype([
        ('sample_id', '<i8'),
        ('row_id', '<i8'),
        ('col_id', '<i8')
    ])

    if len(pixels) > 0:
        pixel_array = np.array(pixels, dtype=[
            ('sample_id', '<i8'), ('row_id', '<i8'), ('col_id', '<i8')
        ])
    else:
        pixel_array = np.array([], dtype=dtype)

    npz_path = os.path.join(output_dir, "data", "pixels.npz")
    np.savez(npz_path, pixels=pixel_array)
    print(f"Saved pixels.npz: {len(pixel_array)} pixel coordinates")

    # 2. Tạo submission.csv với đúng format mới: id, target (target không dùng để chấm)
    if os.path.exists(test_csv_path):
        test_df = pd.read_csv(test_csv_path)
        if 'id' not in test_df.columns:
            raise ValueError(f"File test CSV thiếu cột 'id': {test_csv_path}")
        sub_df = pd.DataFrame({
            'id': test_df['id'].astype(np.int64),
            'target': [''] * len(test_df)
        })
    else:
        # Fallback nếu không tìm thấy test CSV
        unique_ids = sorted(set(p[0] for p in pixels)) if pixels else []
        sub_df = pd.DataFrame({
            'id': np.array(unique_ids, dtype=np.int64),
            'target': [''] * len(unique_ids)
        })

    sub_df.to_csv(os.path.join(output_dir, "submission.csv"), index=False)

    print(f"Saved submission.csv to {output_dir}/")

    # Thống kê
    if len(pixels) > 0:
        sample_ids = [p[0] for p in pixels]
        unique_samples = set(sample_ids)
        print(f"Unique samples with predictions: {len(unique_samples)}")
        print(f"Avg pixels per sample: {len(pixels) / len(unique_samples):.0f}")


def main():
    # Load configs
    with open("configs/data.yaml", "r") as f:
        data_cfg = yaml.safe_load(f)
    with open("configs/train.yaml", "r") as f:
        train_cfg = yaml.safe_load(f)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load best model
    # checkpoint_path = os.path.join(train_cfg["save_dir"], "best.pt")
    checkpoint_path = "./checkpoints/multihead/best.pt"
    model = load_model(checkpoint_path, device)

    test_csv_path = data_cfg.get("test_csv", "data/test.csv")

    # Build map sample_id -> max_width để lọc col_id hợp lệ khi xuất pixels
    max_width_map = None
    if os.path.exists(test_csv_path):
        test_meta_df = pd.read_csv(test_csv_path)
        if 'id' in test_meta_df.columns and 'max_width' in test_meta_df.columns:
            max_width_map = dict(zip(test_meta_df['id'].astype(int), test_meta_df['max_width'].astype(int)))

    # Create test dataloader
    print("\nLoading test data...")
    test_loader = create_test_dataloader(
        test_csv=test_csv_path,
        batch_size=data_cfg.get("batch_size", 16),
        fixed_height=data_cfg.get("fixed_height", 32),
        font_size=data_cfg.get("font_size", 24),
        font_path=data_cfg.get("font_path", None),
    )

    # Run inference
    print("\nRunning inference...")
    pixels = run_inference(
        model,
        test_loader,
        device,
        threshold=0.5,
        visual_dir="inference_visuals",
        image_height=data_cfg.get("fixed_height", 32),
        max_width_map=max_width_map,
    )

    # Save submission
    output_dir = "submission_output"
    save_submission(pixels, output_dir, test_csv_path=test_csv_path)

    print(f"\n=== Inference hoàn tất! ===")
    print(f"Output: {output_dir}/submission.csv + {output_dir}/data/pixels.npz")
    print(f"Total pixel predictions: {len(pixels)}")


if __name__ == "__main__":
    main()
