import json
import torch
import smplx
import numpy as np
import pyvista as pv
import trimesh


# -----------------------------
# CONFIG
# -----------------------------
JSON_PATH = "/home/visvam/daNewFolder/college_files/projects/cip_project/final_codes/3D_reconstruction/core/sample_params.json"
SMPLX_MODEL_DIR = "models/smplx"   # folder containing SMPLX_NEUTRAL.npz etc.
DEVICE = "cpu"                    # "cuda" if available
EXPORT_MESH = True
EXPORT_PATH = "smplx_output.obj"

# -----------------------------
# LOAD JSON
# -----------------------------
with open(JSON_PATH, "r") as f:
    params = json.load(f)

device = torch.device(DEVICE)


# -----------------------------
# CREATE SMPL-X MODEL
# -----------------------------
model = smplx.create(
    SMPLX_MODEL_DIR,
    model_type="smplx",
    gender=params.get("gender", "neutral"),
    use_pca=False,            # REQUIRED for axis-angle hands
    flat_hand_mean=True,
    batch_size=1
).to(device)


# -----------------------------
# HELPER
# -----------------------------
def get_tensor(key, shape):
    if key not in params:
        return torch.zeros(shape, device=device)
    return torch.tensor(
        params[key], dtype=torch.float32, device=device
    ).view(*shape)


# -----------------------------
# FORWARD PASS
# -----------------------------
output = model(
    betas=get_tensor("betas", (1, -1)),
    global_orient=get_tensor("global_orient", (1, 3)),
    body_pose=get_tensor("body_pose", (1, -1)),
    left_hand_pose=get_tensor("left_hand_pose", (1, -1)),
    right_hand_pose=get_tensor("right_hand_pose", (1, -1)),
    jaw_pose=get_tensor("jaw_pose", (1, 3)),
    expression=get_tensor("expression", (1, -1)),
    transl=get_tensor("transl", (1, 3))
)

vertices = output.vertices[0].detach().cpu().numpy()
faces = model.faces


# -----------------------------
# PYVISTA VISUALIZATION
# -----------------------------
faces_pv = np.hstack([
    np.full((faces.shape[0], 1), 3),
    faces
]).astype(np.int64)

pv_mesh = pv.PolyData(vertices, faces_pv)

plotter = pv.Plotter()
plotter.add_mesh(pv_mesh, color="lightgray", smooth_shading=True)
plotter.add_axes()
plotter.show()


# -----------------------------
# TRIMESH VISUALIZATION
# -----------------------------
tri_mesh = trimesh.Trimesh(
    vertices=vertices,
    faces=faces,
    process=False
)

tri_mesh.show()


# -----------------------------
# EXPORT (OPTIONAL)
# -----------------------------
if EXPORT_MESH:
    tri_mesh.export(EXPORT_PATH)
    print("Exported mesh to:", EXPORT_PATH)
