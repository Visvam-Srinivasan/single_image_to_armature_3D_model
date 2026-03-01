"""
SMPL-X Parameter JSON → Joint Coordinates JSON Converter
=========================================================
Converts an SMPL-X parameter JSON file into a joint-coordinate JSON file
whose format matches the reference output (one entry per named joint,
value = [x, y, z] in metres).

Requirements
------------
    pip install smplx torch numpy

SMPL-X model files must be downloaded from https://smpl-x.is.tue.mpg.de/
and placed in a directory like:
    ./models/smplx/SMPLX_MALE.npz
    ./models/smplx/SMPLX_FEMALE.npz
    ./models/smplx/SMPLX_NEUTRAL.npz

Usage
-----
    python smplx_to_joints.py --input params.json --output joints.json --model_path ./models
"""

import argparse
import json
import sys
import numpy as np

# ---------------------------------------------------------------------------
# SMPL-X joint index → output name mapping (127 joints in SMPL-X)
# Indices verified against the official SMPL-X joint regressor order.
# ---------------------------------------------------------------------------
JOINT_MAP = {
    0:  "root",
    1:  "pelvis",
    2:  "left_hip",
    3:  "right_hip",
    4:  "spine1",
    5:  "left_knee",
    6:  "right_knee",
    7:  "spine2",
    8:  "left_ankle",
    9:  "right_ankle",
    10: "spine3",
    11: "left_foot",
    12: "right_foot",
    13: "neck",
    14: "left_collar",
    15: "right_collar",
    16: "head",
    17: "left_shoulder",
    18: "right_shoulder",
    19: "left_elbow",
    20: "right_elbow",
    21: "left_wrist",
    22: "right_wrist",
    # Left hand
    23: "left_index1",
    24: "left_index2",
    25: "left_index3",
    26: "left_middle1",
    27: "left_middle2",
    28: "left_middle3",
    29: "left_pinky1",
    30: "left_pinky2",
    31: "left_pinky3",
    32: "left_ring1",
    33: "left_ring2",
    34: "left_ring3",
    35: "left_thumb1",
    36: "left_thumb2",
    37: "left_thumb3",
    # Right hand
    38: "right_index1",
    39: "right_index2",
    40: "right_index3",
    41: "right_middle1",
    42: "right_middle2",
    43: "right_middle3",
    44: "right_pinky1",
    45: "right_pinky2",
    46: "right_pinky3",
    47: "right_ring1",
    48: "right_ring2",
    49: "right_ring3",
    50: "right_thumb1",
    51: "right_thumb2",
    52: "right_thumb3",
    # Face landmarks (SMPL-X extra joints)
    53: "jaw",
    54: "left_eye_smplhf",
    55: "right_eye_smplhf",
}


def flatten(x):
    """Recursively flatten nested lists / numpy arrays to a 1-D numpy array."""
    arr = np.array(x, dtype=np.float32)
    return arr.flatten()


def load_params(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def build_joints(params: dict, model_path: str) -> dict:
    try:
        import torch
        import smplx
    except ImportError as e:
        sys.exit(
            f"[ERROR] Missing dependency: {e}\n"
            "Install with:  pip install smplx torch"
        )

    gender     = params.get("gender", "neutral")
    betas_raw  = params.get("betas", [0.0] * 10)
    expr_raw   = params.get("expression", [0.0] * 10)
    transl_raw = params.get("transl", [0.0, 0.0, 0.0])

    betas      = flatten(betas_raw)
    expression = flatten(expr_raw)
    transl     = flatten(transl_raw)

    num_betas = len(betas)
    num_expr  = len(expression)

    # ------------------------------------------------------------------ model
    model = smplx.create(
        model_path=model_path,
        model_type="smplx",
        gender=gender,
        num_betas=num_betas,
        num_expression_coeffs=num_expr,
        use_pca=False,          # full per-joint hand pose (15 joints × 3 DoF)
        flat_hand_mean=True,    # zero hand pose = flat/relaxed hand
    )

    def t(arr, shape=None):
        """Numpy → torch float tensor, optionally reshaped."""
        a = np.array(arr, dtype=np.float32)
        if shape:
            a = a.reshape(shape)
        return torch.from_numpy(a).unsqueeze(0)   # add batch dim

    # ------------------------------------------------------------------ poses
    global_orient   = t(flatten(params.get("global_orient",   [0.0]*3)),      (1, 3))
    body_pose       = t(flatten(params.get("body_pose",        [[0.0]*3]*21)), (1, 63))
    jaw_pose        = t(flatten(params.get("jaw_pose",         [0.0]*3)),      (1, 3))
    left_hand_pose  = t(flatten(params.get("left_hand_pose",   [[0.0]*3]*15)), (1, 45))
    right_hand_pose = t(flatten(params.get("right_hand_pose",  [[0.0]*3]*15)), (1, 45))
    leye_pose       = t(np.zeros(3),  (1, 3))
    reye_pose       = t(np.zeros(3),  (1, 3))

    # ---------------------------------------------------------------- forward
    with torch.no_grad():
        output = model(
            betas           = torch.from_numpy(betas).float().unsqueeze(0),
            expression      = torch.from_numpy(expression).float().unsqueeze(0),
            transl          = torch.from_numpy(transl).float().unsqueeze(0),
            global_orient   = global_orient,
            body_pose       = body_pose,
            jaw_pose        = jaw_pose,
            left_hand_pose  = left_hand_pose,
            right_hand_pose = right_hand_pose,
            leye_pose       = leye_pose,
            reye_pose       = reye_pose,
            return_verts    = False,
        )

    joints = output.joints.detach().cpu().numpy()[0]   # (N, 3)

    # -------------------------------------------------------- build output dict
    result = {}
    for idx, name in JOINT_MAP.items():
        if idx < len(joints):
            result[name] = joints[idx].tolist()
        else:
            print(f"[WARN] Joint index {idx} ({name}) out of range — skipping.")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Convert SMPL-X parameter JSON → joint coordinates JSON"
    )
    parser.add_argument(
        "--input",  "-i",
        required=True,
        help="Path to input SMPL-X parameter JSON file"
    )
    parser.add_argument(
        "--output", "-o",
        default="joints_output.json",
        help="Path for the output joint coordinates JSON (default: joints_output.json)"
    )
    parser.add_argument(
        "--model_path", "-m",
        default="./models",
        help="Directory containing SMPL-X model files (default: ./models)"
    )
    args = parser.parse_args()

    print(f"[INFO] Loading parameters from: {args.input}")
    params = load_params(args.input)

    print(f"[INFO] Building SMPL-X model  ({params.get('gender','neutral')}, "
          f"betas={len(params.get('betas',[]))}) …")
    joints = build_joints(params, args.model_path)

    with open(args.output, "w") as f:
        json.dump(joints, f, indent=4)

    print(f"[INFO] Saved {len(joints)} joints → {args.output}")


if __name__ == "__main__":
    main()