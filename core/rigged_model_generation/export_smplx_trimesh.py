#!/usr/bin/env python3
"""
Export SMPL-X mesh in strict T-pose using only betas (shape).
No Blender required.
"""

import os
import sys
import json
import numpy as np
import torch
import smplx
import trimesh


def main():

    if len(sys.argv) < 4:
        print("Usage:")
        print("python export_smplx_trimesh.py model_folder params.json output.obj")
        sys.exit(1)

    model_folder = sys.argv[1]     # Path to SMPL-X models directory
    json_path = sys.argv[2]        # JSON containing betas
    out_obj = sys.argv[3]          # Output OBJ file

    device = torch.device("cpu")

    # -------------------------
    # Load parameters JSON
    # -------------------------
    with open(json_path, "r") as f:
        data = json.load(f)

    gender = data.get("gender", "neutral")
    betas = np.array(data.get("betas", [0.0]*10), dtype=np.float32)

    if betas.ndim == 1:
        betas = betas[None, :]  # Add batch dimension

    betas_tensor = torch.tensor(betas, dtype=torch.float32, device=device)

    # -------------------------
    # Create SMPL-X model
    # -------------------------
    model = smplx.create(
        model_folder,
        model_type="smplx",
        gender=gender,
        use_pca=False,
        batch_size=1,
        num_betas=betas.shape[1],
    ).to(device)

    # -------------------------
    # Force STRICT T-POSE
    # -------------------------
    output = model(
        betas=betas_tensor,
        body_pose=torch.zeros((1, 63), device=device),
        global_orient=torch.zeros((1, 3), device=device),
        left_hand_pose=torch.zeros((1, 45), device=device),
        right_hand_pose=torch.zeros((1, 45), device=device),
        jaw_pose=torch.zeros((1, 3), device=device),
        leye_pose=torch.zeros((1, 3), device=device),
        reye_pose=torch.zeros((1, 3), device=device),
        expression=torch.zeros((1, 10), device=device),
        transl=torch.zeros((1, 3), device=device),
    )

    vertices = output.vertices.detach().cpu().numpy().squeeze()
    faces = model.faces

    # -------------------------
    # Create mesh with trimesh
    # -------------------------
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    mesh.export(out_obj)

    print("Exported T-pose mesh to:", out_obj)


if __name__ == "__main__":
    main()