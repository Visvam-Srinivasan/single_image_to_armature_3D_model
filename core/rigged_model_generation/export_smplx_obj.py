#!/usr/bin/env python3
"""
Export SMPL-X model to OBJ from a JSON parameter file.
"""

import bpy
import sys
import os
import json
import importlib

try:
    import numpy as np
except Exception:
    np = None


def eprint(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)


def parse_args():
    argv = sys.argv
    if "--" in argv:
        idx = argv.index("--")
        args = argv[idx + 1 :]
    else:
        args = []
    return args


def ensure_addon_module():
    try:
        import smplx_blender_addon as smplx_mod
        return smplx_mod
    except Exception:
        addon_dir = os.path.dirname(__file__)
        if addon_dir not in sys.path:
            sys.path.append(addon_dir)
        smplx_mod = importlib.import_module('smplx_blender_addon')
        try:
            if hasattr(smplx_mod, 'register'):
                smplx_mod.register()
        except Exception:
            pass
        return smplx_mod


def to_np(v, shape=None):
    if np is None:
        return v
    a = np.asarray(v)
    if shape is not None:
        return a.reshape(shape)
    return a


def main():
    args = parse_args()
    if len(args) < 1:
        eprint("Usage: blender --background --python export_smplx_obj.py -- params.json [out.obj]")
        return

    json_path = os.path.abspath(args[0])
    base_name = os.path.basename(args[1]) if len(args) > 1 else os.path.splitext(os.path.basename(json_path))[0]
    out_obj = os.path.abspath(os.path.join("outputs", base_name + ".obj"))
    os.makedirs(os.path.dirname(out_obj), exist_ok=True)
    if not os.path.exists(json_path):
        eprint("Parameter file not found:", json_path)
        return

    smplx_mod = ensure_addon_module()

    if not hasattr(bpy.context.window_manager, 'smplx_tool'):
        try:
            smplx_mod.register()
        except Exception:
            pass

    wm = bpy.context.window_manager.smplx_tool

    with open(json_path, 'r') as f:
        data = json.load(f)

    gender = data.get('gender', data.get('model', 'neutral'))
    if isinstance(gender, str) and gender.startswith('smplx_'):
        gender = gender.split('_')[-1]

    wm.smplx_gender = gender
    wm.smplx_handpose = data.get('hand_pose', data.get('handpose', 'relaxed'))
    wm.smplx_uv = 'UV_2023' if data.get('uv', '2023') == '2023' else 'UV_2021'
    wm.smplx_corrective_poseshapes = data.get('corrective_poseshapes', False)

    bpy.ops.scene.smplx_add_gender()

    obj = bpy.context.view_layer.objects.active
    if obj is None or obj.type != 'MESH':
        eprint('Failed to add SMPL-X mesh')
        return

    armature = obj.parent

    betas = data.get('betas', data.get('shape', []))
    if betas:
        bpy.ops.object.mode_set(mode='OBJECT')
        for i, b in enumerate(betas):
            name = f"Shape{i:03d}"
            if name in obj.data.shape_keys.key_blocks:
                obj.data.shape_keys.key_blocks[name].value = float(b)
        bpy.ops.object.smplx_update_joint_locations('EXEC_DEFAULT')

    def pick(*keys):
        for k in keys:
            if k in data:
                return data[k]
        return None

    global_orient = pick('global_orient', 'globalOrient', 'global_rotation')
    body_pose = pick('body_pose', 'bodyPose', 'body_pose_flat', 'body')
    jaw_pose = pick('jaw_pose', 'jaw')
    left_hand = pick('left_hand_pose', 'leftHandPose', 'left_hand')
    right_hand = pick('right_hand_pose', 'rightHandPose', 'right_hand')
    translation = pick('transl', 'trans', 'translation')
    expression = pick('expression', 'expr')

    hand_relaxed = None
    try:
        addon_dir = os.path.dirname(smplx_mod.__file__)
        hand_npz = os.path.join(addon_dir, 'data', 'smplx_handposes.npz')
        if os.path.exists(hand_npz) and np is not None:
            with np.load(hand_npz, allow_pickle=True) as npz:
                hand_poses = npz['hand_poses'].item()
                left_rel, right_rel = hand_poses.get('relaxed', (None, None))
                if left_rel is not None and right_rel is not None:
                    hand_relaxed = np.concatenate((left_rel, right_rel)).reshape(-1, 3)
    except Exception:
        hand_relaxed = None

    if global_orient is not None:
        smplx_mod.set_pose_from_rodrigues(armature, 'pelvis', to_np(global_orient).reshape(3))

    if body_pose is not None:
        bp = to_np(body_pose).reshape(-1, 3)
        for i in range(min(len(bp), smplx_mod.NUM_SMPLX_BODYJOINTS)):
            bone_name = smplx_mod.SMPLX_JOINT_NAMES[i + 1]
            smplx_mod.set_pose_from_rodrigues(armature, bone_name, bp[i])

    if jaw_pose is not None:
        smplx_mod.set_pose_from_rodrigues(armature, 'jaw', to_np(jaw_pose).reshape(3))

    hand_start = 1 + smplx_mod.NUM_SMPLX_BODYJOINTS + 3
    if left_hand is not None:
        lh = to_np(left_hand).reshape(-1, 3)
        for i in range(min(len(lh), smplx_mod.NUM_SMPLX_HANDJOINTS)):
            bone = smplx_mod.SMPLX_JOINT_NAMES[hand_start + i]
            ref = hand_relaxed[i] if hand_relaxed is not None else None
            smplx_mod.set_pose_from_rodrigues(armature, bone, lh[i], ref)

    if right_hand is not None:
        rh = to_np(right_hand).reshape(-1, 3)
        for i in range(min(len(rh), smplx_mod.NUM_SMPLX_HANDJOINTS)):
            bone = smplx_mod.SMPLX_JOINT_NAMES[hand_start + smplx_mod.NUM_SMPLX_HANDJOINTS + i]
            ref = hand_relaxed[smplx_mod.NUM_SMPLX_HANDJOINTS + i] if hand_relaxed is not None else None
            smplx_mod.set_pose_from_rodrigues(armature, bone, rh[i], ref)

    if translation is not None:
        t = to_np(translation).reshape(3)
        armature.location = (float(t[0]), -float(t[2]), float(t[1]))

    if wm.smplx_corrective_poseshapes:
        bpy.ops.object.smplx_set_poseshapes('EXEC_DEFAULT')

    if expression is not None:
        for i, exp in enumerate(expression):
            name = f"Exp{i:03d}"
            if name in obj.data.shape_keys.key_blocks:
                obj.data.shape_keys.key_blocks[name].value = float(exp)

    # Force evaluation of the scene so modifiers are updated
    bpy.context.view_layer.update()

    # Deselect all, then select ONLY the mesh object
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    if bpy.app.version >= (3, 2, 0):
        bpy.ops.wm.obj_export(
            filepath=out_obj,
            export_selected_objects=True,
            apply_modifiers=True
        )
    else:
        bpy.ops.export_scene.obj(
            filepath=out_obj,
            use_selection=True,
            use_mesh_modifiers=True
        )

    print('Exported static posed OBJ:', out_obj)

if __name__ == '__main__':
    main()