import os
import yaml
import numpy as np
import torch
from tqdm import tqdm

from models.baseline import BaselineModel
from util.data_loader import create_test_dataloader


def load_model(checkpoint_path, device):
    """Load model từ checkpoint best.pt"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint không tìm thấy: {checkpoint_path}")

    model = BaselineModel().to(device)

    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # best.pt lưu bằng torch.save(model.state_dict(), ...) 
    model.load_state_dict(checkpoint)
    model.eval()
    print("Model loaded successfully!")
    return model


@torch.no_grad()
def run_inference(model, test_loader, device, threshold=0.5):
    """
    Chạy inference trên test set.
    Trả về list các (sample_id, row_id, col_id) cho các pixel dự đoán "sáng" (text).
    """
    model.eval()
    all_pixels = []  # List of (sample_id, row_id, col_id)

    pbar = tqdm(test_loader, desc="Inference")
    for batch in pbar:
        contexts, ids = batch
        contexts = contexts.to(device)

        # Forward pass: model dự đoán ảnh target từ ảnh context
        outputs = model(contexts)  # (B, 1, H, W), giá trị [0, 1] sau Sigmoid

        # Threshold: pixel > 0.5 → text (sáng), <= 0.5 → background (tối)
        predictions = (outputs > threshold).squeeze(1).cpu().numpy()  # (B, H, W)
        sample_ids = ids.cpu().numpy()

        # Trích xuất tọa độ pixel "sáng" cho mỗi sample
        for i in range(len(sample_ids)):
            sid = int(sample_ids[i])
            pred = predictions[i]  # (H, W)

            # Tìm tọa độ (row, col) của các pixel sáng
            rows, cols = np.where(pred > 0)

            for r, c in zip(rows, cols):
                all_pixels.append((sid, int(r), int(c)))

        pbar.set_postfix({"pixels_found": len(all_pixels)})

    return all_pixels


def save_submission(pixels, output_dir="submission_output"):
    """
    Lưu kết quả submission theo format:
    - submission.csv: mapping id → sample_id
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

    # 2. Tạo submission.csv (copy format từ sample_submission)
    import pandas as pd
    sample_sub_path = "sample_submission/submission.csv"
    if os.path.exists(sample_sub_path):
        sub_df = pd.read_csv(sample_sub_path)
        sub_df.to_csv(os.path.join(output_dir, "submission.csv"), index=False)
    else:
        # Tạo mới nếu không có sample
        unique_ids = sorted(set(p[0] for p in pixels)) if pixels else []
        sub_df = pd.DataFrame({
            'id': range(len(unique_ids)),
            'sample_id': unique_ids
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
    checkpoint_path = os.path.join(train_cfg["save_dir"], "best.pt")
    model = load_model(checkpoint_path, device)

    # Create test dataloader
    print("\nLoading test data...")
    test_loader = create_test_dataloader(
        test_csv=data_cfg.get("test_csv", "data/test.csv"),
        batch_size=data_cfg.get("batch_size", 16),
        fixed_height=data_cfg.get("fixed_height", 32),
        font_size=data_cfg.get("font_size", 24),
        font_path=data_cfg.get("font_path", None),
    )

    # Run inference
    print("\nRunning inference...")
    pixels = run_inference(model, test_loader, device, threshold=0.5)

    # Save submission
    output_dir = "submission_output"
    save_submission(pixels, output_dir)

    print(f"\n=== Inference hoàn tất! ===")
    print(f"Output: {output_dir}/submission.csv + {output_dir}/data/pixels.npz")
    print(f"Total pixel predictions: {len(pixels)}")


if __name__ == "__main__":
    main()
