#!/usr/bin/env python3
import bpy
import sys
import os
import json
import importlib
from mathutils import Matrix

try:
    import numpy as np
except Exception:
    np = None

def parse_args():
    argv = sys.argv
    return argv[argv.index("--") + 1 :] if "--" in argv else []

def ensure_addon_module():
    try:
        import smplx_blender_addon as smplx_mod
        return smplx_mod
    except Exception:
        addon_dir = os.path.dirname(__file__)
        if addon_dir not in sys.path:
            sys.path.append(addon_dir)
        smplx_mod = importlib.import_module('smplx_blender_addon')
        return smplx_mod

def to_np(v, shape=None):
    if np is None:
        return v
    a = np.asarray(v)
    return a.reshape(shape) if shape is not None else a

def main():
    args = parse_args()
    if len(args) < 1:
        return

    json_path = os.path.abspath(args[0])

    if not os.path.exists(json_path):
        print("Parameter file not found:", json_path)
        return

    # 🔹 Create output filename automatically
    input_dir = os.path.dirname(json_path)
    input_base = os.path.splitext(os.path.basename(json_path))[0]
    out_joints_json = os.path.join(input_dir, input_base + "_joints.json")

    smplx_mod = ensure_addon_module()
    if not hasattr(bpy.context.window_manager, 'smplx_tool'):
        smplx_mod.register()

    wm = bpy.context.window_manager.smplx_tool
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 1. Setup Model Metadata
    gender = data.get('gender', 'neutral')
    wm.smplx_gender = gender.split('_')[-1] if 'smplx' in gender else gender
    bpy.ops.scene.smplx_add_gender()

    obj = bpy.context.view_layer.objects.active
    armature = obj.parent

    armature.rotation_euler = (0, 0, 0)

    # 2. Apply Shape
    betas = data.get('betas', data.get('shape', []))
    if betas:
        for i, b in enumerate(betas):
            name = f"Shape{i:03d}"
            if name in obj.data.shape_keys.key_blocks:
                obj.data.shape_keys.key_blocks[name].value = float(b)
        
        bpy.context.view_layer.update()
        bpy.ops.object.smplx_update_joint_locations('EXEC_DEFAULT')

    # 3. Apply Pose
    def get_val(*keys):
        for k in keys:
            if k in data:
                return data[k]
        return None

    g_orient = get_val('global_orient', 'global_rotation')
    b_pose   = get_val('body_pose', 'body')
    l_hand   = get_val('left_hand_pose', 'left_hand')
    r_hand   = get_val('right_hand_pose', 'right_hand')
    transl   = get_val('transl', 'translation')

    if g_orient is not None:
        smplx_mod.set_pose_from_rodrigues(
            armature,
            'pelvis',
            to_np(g_orient).reshape(3)
        )

    if b_pose is not None:
        bp = to_np(b_pose).reshape(-1, 3)
        for i in range(min(len(bp), 21)):
            smplx_mod.set_pose_from_rodrigues(
                armature,
                smplx_mod.SMPLX_JOINT_NAMES[i + 1],
                bp[i]
            )

    h_idx = 22

    if l_hand is not None:
        lh = to_np(l_hand).reshape(-1, 3)
        for i in range(min(len(lh), 15)):
            smplx_mod.set_pose_from_rodrigues(
                armature,
                smplx_mod.SMPLX_JOINT_NAMES[h_idx + i],
                lh[i]
            )

    if r_hand is not None:
        rh = to_np(r_hand).reshape(-1, 3)
        for i in range(min(len(rh), 15)):
            smplx_mod.set_pose_from_rodrigues(
                armature,
                smplx_mod.SMPLX_JOINT_NAMES[h_idx + 15 + i],
                rh[i]
            )

    if transl is not None:
        t = to_np(transl).reshape(3)
        armature.location = (t[0], t[1], t[2])

    bpy.context.view_layer.update()

    # 5. Export
    joint_locations = {}
    for bone in armature.data.bones:
        pos = armature.matrix_world @ bone.head_local
        joint_locations[bone.name] = [pos.x, pos.y, pos.z]

    with open(out_joints_json, 'w') as jf:
        json.dump(joint_locations, jf, indent=4)

    print("Joint file saved to:", out_joints_json)

if __name__ == '__main__':
    main()