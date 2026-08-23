from pathlib import Path

from pipeline.video_reader import discover_videos, infer_label


def test_infer_label_from_dataset_folder() -> None:
    root = Path("data")
    assert infer_label(root / "normal" / "normal-1.mp4", root) == "normal"
    assert infer_label(root / "shoplifting" / "shoplifting-1.mp4", root) == "shoplifting"


def test_discover_videos_finds_dataset_clips() -> None:
    videos = discover_videos("data")
    assert any(path.parent.name == "normal" for path in videos)
    assert any(path.parent.name == "shoplifting" for path in videos)
