import json

import numpy as np

from datasets.pose_json import camera_of, gt_stem, load_segments


def _frame(nan_joint=None):
    kp = np.tile([100.0, 200.0, 0.9], (17, 1))
    if nan_joint is not None:
        kp[nan_joint] = np.nan
    return {"keypoints": kp.ravel().tolist()}


def test_nan_joints_become_missing_and_gaps_split(tmp_path):
    frames = {str(i): _frame(nan_joint=9 if i == 10 else None) for i in range(40)}
    frames.update({str(i): _frame() for i in range(100, 140)})  # gap > max_gap -> new segment
    path = tmp_path / "1_clip.json"
    path.write_text(json.dumps({"7": frames}))
    segs = load_segments(path, min_len=32)
    assert len(segs) == 2
    kp = segs[0]["keypoints"]
    assert np.isfinite(kp).all()
    assert kp[10, 9, 2] == 0.0  # NaN joint is marked missing
    assert np.allclose(kp[10, 9, :2], [100.0, 200.0])  # coordinates filled from neighbours


def test_name_helpers():
    assert camera_of("01_0222_alphapose_tracked_person") == "1"
    assert camera_of("4_042820_2") == "4"
    assert gt_stem("01_0222_alphapose_tracked_person") == "01_0222"
