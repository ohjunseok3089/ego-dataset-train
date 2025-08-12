from src.data_loader import EgoComDataset

if __name__ == "__main__":
    # Example usage; update path to your EGOCOM root
    dataset = EgoComDataset(
        data_dir="/path/to/EGOCOM",
        sample_every_n=5,
        max_frames=120,
        return_annotations=True,
    )
    print(f"Found {len(dataset)} parts")
    if len(dataset) > 0:
        item = dataset[0]
        video = item["video"]
        print(
            f"Video tensor: shape={tuple(video.shape)}, dtype={video.dtype}, fps={item['fps']}"
        )
        print("Keys:", list(item.keys()))
