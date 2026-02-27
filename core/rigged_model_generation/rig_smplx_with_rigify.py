#!/usr/bin/env python3
"""
Rig an SMPL-X OBJ mesh using Blender's Rigify addon.
- Positions metarig bones from SMPL-X joint positions
- Separates loose mesh islands automatically
- Skins each island (face, eyes, teeth, hands, breasts) 100% to the correct bone
- Skins the main body with automatic weights
- Renames deform bones for Unity Humanoid auto-mapping
- Reparents DEF spine chain so Unity can walk the parent hierarchy correctly
- Fixed foot/toe bone direction (forward = -Y in Blender Z-up space)
- Fixed Shin/Foot morphing by placing Rigify heel bones.
- Fixed Toes pointing up by keeping toe vectors horizontal.

Usage (headless Blender):
  blender --background --python rig_smplx_with_rigify.py -- joints.json mesh.obj [out.blend]
"""

import bpy
import sys
import os
import json
import mathutils
import traceback


def flush(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def eprint(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)
    sys.stderr.flush()


def parse_args():
    argv = sys.argv
    if "--" in argv:
        return argv[argv.index("--") + 1:]
    return []


# ─────────────────────────────────────────────────────────────────────────────
#  Enable Rigify
# ─────────────────────────────────────────────────────────────────────────────

def enable_rigify():
    flush("Enabling Rigify addon...")
    try:
        bpy.ops.preferences.addon_enable(module='rigify')
        flush("  Rigify enabled via preferences operator.")
    except Exception as e:
        flush(f"  preferences.addon_enable failed ({e}), trying addon_utils...")
        try:
            import addon_utils as au
            loaded, _ = au.check('rigify')
            if not loaded:
                au.enable('rigify', default_set=True, persistent=True)
            flush("  Rigify enabled via addon_utils.")
        except Exception as e2:
            eprint(f"  ERROR: Could not enable Rigify: {e2}")
            return False

    bpy.context.view_layer.update()

    if not hasattr(bpy.ops.object, 'armature_human_metarig_add'):
        eprint("  ERROR: armature_human_metarig_add not found after enabling Rigify.")
        return False

    flush("  Rigify operators OK.")
    return True


# ─────────────────────────────────────────────────────────────────────────────
#  Bone positioning helpers
# ─────────────────────────────────────────────────────────────────────────────

def jvec(joints, name):
    if name not in joints:
        return None
    return mathutils.Vector(joints[name])


def safe_set_bone(arm, name, head_world, tail_world):
    if name not in arm.edit_bones:
        return

    arm_obj = bpy.context.object  # metarig object
    inv = arm_obj.matrix_world.inverted()

    head = inv @ head_world
    tail = inv @ tail_world

    b = arm.edit_bones[name]
    b.head = head

    if (tail - head).length < 1e-4:
        tail = head + mathutils.Vector((0, 0.001, 0))

    b.tail = tail


def extend(origin, direction_vec, length=0.08):
    return origin + direction_vec.normalized() * length


def position_metarig_bones(metarig, joints):
    arm = metarig.data

    def j(name, fallback=None):
        v = jvec(joints, name)
        return v if v is not None else fallback

    pelvis     = j("pelvis")
    l_hip      = j("left_hip")
    r_hip      = j("right_hip")
    l_knee     = j("left_knee")
    r_knee     = j("right_knee")
    l_ankle    = j("left_ankle")
    r_ankle    = j("right_ankle")

    # SMPL-X exports with Y-up→Z-up swap, so body faces -Y.
    l_foot     = j("left_foot")
    r_foot     = j("right_foot")

    spine1     = j("spine1")
    spine2     = j("spine2")
    spine3     = j("spine3")
    neck       = j("neck")
    head_jnt   = j("head")
    l_collar   = j("left_collar",  spine3)
    r_collar   = j("right_collar", spine3)
    l_shoulder = j("left_shoulder")
    r_shoulder = j("right_shoulder")
    l_elbow    = j("left_elbow")
    r_elbow    = j("right_elbow")
    l_wrist    = j("left_wrist")
    r_wrist    = j("right_wrist")
    l_breast   = j("left_breast")
    r_breast   = j("right_breast")

    if pelvis is None or spine1 is None:
        eprint("  CRITICAL: pelvis or spine1 missing from joints JSON.")
        return

    spine_dir = (neck - pelvis).normalized() if (neck and pelvis) else mathutils.Vector((0, 0, 1))
    head_top  = None
    if head_jnt and neck:
        dist = max((head_jnt - neck).length * 0.9, 0.15)
        head_top = head_jnt + spine_dir * dist
    elif head_jnt:
        head_top = head_jnt + mathutils.Vector((0, 0, 0.15))

    # FIX 1: Toe bone MUST be horizontal in Rigify. If it points downwards into the floor, 
    # the generated IK foot control will tilt, causing toes to bend upwards when snapped flat.
    if l_foot and l_ankle:
        l_toe_dir = mathutils.Vector((l_foot.x - l_ankle.x, l_foot.y - l_ankle.y, 0)) # Z is strictly 0
        if l_toe_dir.length < 1e-4: l_toe_dir = mathutils.Vector((0, -1, 0))
        l_toe = l_foot + l_toe_dir.normalized() * 0.10
    else:
        l_toe = None

    if r_foot and r_ankle:
        r_toe_dir = mathutils.Vector((r_foot.x - r_ankle.x, r_foot.y - r_ankle.y, 0)) # Z is strictly 0
        if r_toe_dir.length < 1e-4: r_toe_dir = mathutils.Vector((0, -1, 0))
        r_toe = r_foot + r_toe_dir.normalized() * 0.10
    else:
        r_toe = None

    l_hand_tip = extend(l_wrist, l_wrist - l_elbow) if (l_wrist and l_elbow) else None
    r_hand_tip = extend(r_wrist, r_wrist - r_elbow) if (r_wrist and r_elbow) else None

    if pelvis and spine1:        safe_set_bone(arm, "spine",     pelvis,   spine1)
    if spine1 and spine2:        safe_set_bone(arm, "spine.001", spine1,   spine2)
    if spine2 and spine3:        safe_set_bone(arm, "spine.002", spine2,   spine3)
    if spine3 and neck:          safe_set_bone(arm, "spine.003", spine3,   neck)
    if neck and head_jnt:        safe_set_bone(arm, "spine.004", neck,     head_jnt)
    if head_jnt and head_top:    safe_set_bone(arm, "spine.005", head_jnt, head_top)
    if head_top is not None:     safe_set_bone(arm, "spine.006", head_top, head_top + spine_dir * 0.02)

    if l_collar and l_shoulder:  safe_set_bone(arm, "shoulder.L", l_collar, l_shoulder)
    if r_collar and r_shoulder:  safe_set_bone(arm, "shoulder.R", r_collar, r_shoulder)

    if l_shoulder and l_elbow:   safe_set_bone(arm, "upper_arm.L", l_shoulder, l_elbow)
    if r_shoulder and r_elbow:   safe_set_bone(arm, "upper_arm.R", r_shoulder, r_elbow)
    if l_elbow and l_wrist:      safe_set_bone(arm, "forearm.L",   l_elbow,    l_wrist)
    if r_elbow and r_wrist:      safe_set_bone(arm, "forearm.R",   r_elbow,    r_wrist)
    if l_wrist and l_hand_tip:   safe_set_bone(arm, "hand.L",      l_wrist,    l_hand_tip)
    if r_wrist and r_hand_tip:   safe_set_bone(arm, "hand.R",      r_wrist,    r_hand_tip)

    if l_hip and l_knee:         safe_set_bone(arm, "thigh.L", l_hip,   l_knee)
    if r_hip and r_knee:         safe_set_bone(arm, "thigh.R", r_hip,   r_knee)
    if l_knee and l_ankle:       safe_set_bone(arm, "shin.L",  l_knee,  l_ankle)
    if r_knee and r_ankle:       safe_set_bone(arm, "shin.R",  r_knee,  r_ankle)
    if l_ankle and l_foot:       safe_set_bone(arm, "foot.L",  l_ankle, l_foot)
    if r_ankle and r_foot:       safe_set_bone(arm, "foot.R",  r_ankle, r_foot)
    if l_foot and l_toe:         safe_set_bone(arm, "toe.L",   l_foot,  l_toe)
    if r_foot and r_toe:         safe_set_bone(arm, "toe.R",   r_foot,  r_toe)

    # FIX 2: Rigify requires heel bones to be properly placed for foot IK and leg roll.
    # Leaving them at the origin (0,0,0) causes severe shin twisting and foot morphing.
    if l_ankle and l_foot:
        # Place heel behind the ankle (+Y is back) at the vertical level of the toe base (l_foot.z)
        l_heel_head = mathutils.Vector((l_ankle.x, l_ankle.y + 0.05, l_foot.z))
        l_heel_tail = l_heel_head + mathutils.Vector((0, 0.05, 0))
        safe_set_bone(arm, "heel.02.L", l_heel_head, l_heel_tail)

    if r_ankle and r_foot:
        r_heel_head = mathutils.Vector((r_ankle.x, r_ankle.y + 0.05, r_foot.z))
        r_heel_tail = r_heel_head + mathutils.Vector((0, 0.05, 0))
        safe_set_bone(arm, "heel.02.R", r_heel_head, r_heel_tail)

    if l_breast and r_breast:
        safe_set_bone(arm, "breast.L", l_breast, l_breast + mathutils.Vector((0, -0.05, 0)))
        safe_set_bone(arm, "breast.R", r_breast, r_breast + mathutils.Vector((0, -0.05, 0)))
    elif spine3:
        b_l_head = spine3 + mathutils.Vector(( 0.08, -0.06, -0.04))
        b_l_tail = b_l_head + mathutils.Vector((0.0, -0.05, 0.0))
        b_r_head = spine3 + mathutils.Vector((-0.08, -0.06, -0.04))
        b_r_tail = b_r_head + mathutils.Vector((0.0, -0.05, 0.0))
        safe_set_bone(arm, "breast.L", b_l_head, b_l_tail)
        safe_set_bone(arm, "breast.R", b_r_head, b_r_tail)

    flush("  Metarig bones positioned.")


# ─────────────────────────────────────────────────────────────────────────────
#  Strip unwanted metarig bones (face + fingers)
# ─────────────────────────────────────────────────────────────────────────────

BONES_TO_REMOVE_ROOTS = [
    "face",
    "thumb.01.L",    "thumb.01.R",
    "f_index.01.L",  "f_index.01.R",
    "f_middle.01.L", "f_middle.01.R",
    "f_ring.01.L",   "f_ring.01.R",
    "f_pinky.01.L",  "f_pinky.01.R",
    "palm.01.L",     "palm.01.R",
    "palm.02.L",     "palm.02.R",
    "palm.03.L",     "palm.03.R",
    "palm.04.L",     "palm.04.R",
]


def _collect_bone_tree(arm, bone_name, result):
    if bone_name not in arm.edit_bones:
        return
    result.add(bone_name)
    for child in arm.edit_bones[bone_name].children:
        _collect_bone_tree(arm, child.name, result)


def strip_face_and_finger_bones(metarig):
    arm = metarig.data
    to_delete = set()
    for root_name in BONES_TO_REMOVE_ROOTS:
        _collect_bone_tree(arm, root_name, to_delete)
    for b in arm.edit_bones:
        b.select = b.name in to_delete
    deleted = [n for n in to_delete if n in arm.edit_bones]
    if deleted:
        bpy.ops.armature.delete()
        flush(f"  Removed {len(deleted)} face/finger bones from metarig.")
    else:
        flush("  No face/finger bones found to remove.")


# ─────────────────────────────────────────────────────────────────────────────
#  Mesh island utilities
# ─────────────────────────────────────────────────────────────────────────────

def is_smplx_mesh(obj):
    if obj.type != 'MESH':
        return False
    if obj.name.startswith("WGT-"):
        return False
    try:
        _ = obj.visible_get()
    except Exception:
        return False
    if obj.hide_viewport:
        return False
    return True


def mesh_center(obj):
    corners = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
    return sum(corners, mathutils.Vector()) / 8.0


def _skin_to_bone(obj, rig, bone_name):
    """Skin obj 100% to a specific bone via Armature modifier + vertex group."""
    if bone_name is None:
        eprint(f"    [warn] No bone name for '{obj.name}', skipping.")
        return

    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    mat_world = obj.matrix_world.copy()
    obj.parent = rig
    obj.parent_type = 'OBJECT'
    obj.matrix_world = mat_world

    if not any(m.type == 'ARMATURE' for m in obj.modifiers):
        mod = obj.modifiers.new(name="Armature", type='ARMATURE')
        mod.object = rig

    vg = obj.vertex_groups.get(bone_name) or obj.vertex_groups.new(name=bone_name)
    vg.add([v.index for v in obj.data.vertices], 1.0, 'REPLACE')
    flush(f"    Skinned '{obj.name}' → bone '{bone_name}'")


def separate_and_skin_islands(mesh_obj, rig, joints):
    flush("Separating mesh into loose parts...")

    pre_existing_names = {o.name for o in bpy.context.scene.objects if o.type == 'MESH'}

    bpy.ops.object.select_all(action='DESELECT')
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.separate(type='LOOSE')
    bpy.ops.object.mode_set(mode='OBJECT')

    smplx_islands = [
        o for o in bpy.context.scene.objects
        if is_smplx_mesh(o) and (
            o.name == mesh_obj.name or o.name.startswith("SMPLX_Body")
        )
    ]
    if len(smplx_islands) <= 1:
        smplx_islands = [
            o for o in bpy.context.scene.objects
            if is_smplx_mesh(o) and not o.name.startswith("WGT-")
            and (o.name == mesh_obj.name or o.name not in pre_existing_names
                 or o.name.startswith("SMPLX"))
        ]

    flush(f"  Found {len(smplx_islands)} SMPL-X mesh islands.")

    if len(smplx_islands) <= 1:
        flush("  Only one island — no separation needed.")
        return mesh_obj

    def jv(name):
        return mathutils.Vector(joints[name]) if name in joints else None

    head_pos   = jv("head")
    l_wrist    = jv("left_wrist")
    r_wrist    = jv("right_wrist")
    spine2_pos = jv("spine2")
    spine3_pos = jv("spine3")
    chest_pos  = (spine2_pos + spine3_pos) / 2.0 if (spine2_pos and spine3_pos) else spine3_pos

    smplx_islands.sort(key=lambda o: len(o.data.vertices), reverse=True)
    body_obj = smplx_islands[0]
    body_obj.name = "SMPLX_Body"
    flush(f"  Body mesh : {body_obj.name} ({len(body_obj.data.vertices)} verts)")

    HEAD_THRESHOLD   = 0.35
    HAND_THRESHOLD   = 0.25
    BREAST_THRESHOLD = 0.30

    def find_bone(keywords):
        for kw in keywords:
            if kw in rig.pose.bones:
                return kw
        for kw in keywords:
            for b in rig.pose.bones:
                if kw.lower() in b.name.lower():
                    return b.name
        return None

    head_bone   = find_bone(["Head", "DEF-spine.005"])
    l_hand_bone = find_bone(["LeftHand", "DEF-hand.L"])
    r_hand_bone = find_bone(["RightHand", "DEF-hand.R"])
    chest_bone  = find_bone(["UpperChest", "Chest", "DEF-spine.003"])

    flush(f"  Head bone   : {head_bone}")
    flush(f"  L hand bone : {l_hand_bone}")
    flush(f"  R hand bone : {r_hand_bone}")
    flush(f"  Chest bone  : {chest_bone}")

    for obj in smplx_islands[1:]:
        center = mesh_center(obj)
        dist_head  = (center - head_pos).length  if head_pos  else 999
        dist_lh    = (center - l_wrist).length   if l_wrist   else 999
        dist_rh    = (center - r_wrist).length   if r_wrist   else 999
        dist_chest = (center - chest_pos).length if chest_pos else 999
        min_dist   = min(dist_head, dist_lh, dist_rh)

        if dist_head == min_dist and dist_head < HEAD_THRESHOLD:
            flush(f"  → head   : {obj.name} ({len(obj.data.vertices)}v, d={dist_head:.3f}m)")
            _skin_to_bone(obj, rig, head_bone)
        elif dist_lh == min_dist and dist_lh < HAND_THRESHOLD:
            flush(f"  → L hand : {obj.name} ({len(obj.data.vertices)}v, d={dist_lh:.3f}m)")
            _skin_to_bone(obj, rig, l_hand_bone)
        elif dist_rh == min_dist and dist_rh < HAND_THRESHOLD:
            flush(f"  → R hand : {obj.name} ({len(obj.data.vertices)}v, d={dist_rh:.3f}m)")
            _skin_to_bone(obj, rig, r_hand_bone)
        elif dist_chest < BREAST_THRESHOLD:
            flush(f"  → breast : {obj.name} ({len(obj.data.vertices)}v, d_chest={dist_chest:.3f}m)")
            _skin_to_bone(obj, rig, chest_bone)
        else:
            bone_name = find_bone(["UpperChest", "Chest", "DEF-spine.003", "spine"])
            flush(f"  → unknown: {obj.name} → '{bone_name}'")
            _skin_to_bone(obj, rig, bone_name)

    flush("  Island skinning complete.")
    return body_obj


# ─────────────────────────────────────────────────────────────────────────────
#  Fix breast vertex weights on the body mesh
# ─────────────────────────────────────────────────────────────────────────────

def fix_breast_weights(body_obj, rig, joints):
    """
    Reassign breast vertices to the UpperChest/DEF-spine.003 bone.
    Strategy 1: named vertex groups 'breast.L' / 'breast.R'.
    Strategy 2: proximity fallback near chest midpoint.
    """
    target_vg_name = None
    for vg in body_obj.vertex_groups:
        n = vg.name.lower()
        if "upperchest" in n or ("spine.003" in n and "def" in n) or "spine.003" in n:
            target_vg_name = vg.name
            break
    if not target_vg_name:
        flush("  [breast fix] Target vertex group not found, skipping.")
        return

    target_vg = body_obj.vertex_groups[target_vg_name]
    breast_verts = set()

    # Strategy 1 — named vertex groups
    named = [vg for vg in body_obj.vertex_groups if "breast" in vg.name.lower()]
    flush(f"  [breast fix] Named breast groups: {[v.name for v in named]}")
    for vg in named:
        for v in body_obj.data.vertices:
            for g in v.groups:
                if g.group == vg.index and g.weight > 0.01:
                    breast_verts.add(v.index)
    flush(f"  [breast fix] Strategy 1 (named): {len(breast_verts)} verts")

    # Strategy 2 — proximity fallback
    if not breast_verts:
        def jv(name):
            return mathutils.Vector(joints[name]) if name in joints else None
        s2 = jv("spine2")
        s3 = jv("spine3")
        if s2 and s3:
            center = (s2 + s3) / 2.0 + mathutils.Vector((0.0, -0.08, 0.0))
            mw = body_obj.matrix_world
            for v in body_obj.data.vertices:
                if (mw @ v.co - center).length < 0.14:
                    breast_verts.add(v.index)
            flush(f"  [breast fix] Strategy 2 (proximity): {len(breast_verts)} verts")

    if not breast_verts:
        flush("  [breast fix] No breast vertices found, skipping.")
        return

    breast_verts = list(breast_verts)
    for vg in body_obj.vertex_groups:
        if vg.name != target_vg_name:
            try:
                vg.remove(breast_verts)
            except Exception:
                pass
    target_vg.add(breast_verts, 1.0, 'REPLACE')
    flush(f"  [breast fix] {len(breast_verts)} verts reassigned → '{target_vg_name}'")


# ─────────────────────────────────────────────────────────────────────────────
#  Unity Humanoid rename + reparent
# ─────────────────────────────────────────────────────────────────────────────

# Spine chain: (DEF bone name, Unity name, Unity parent name)
# Order is root → tip so parents are renamed before children
UNITY_SPINE_CHAIN = [
    ("DEF-spine",     "Hips",       None),
    ("DEF-spine.001", "Spine",      "Hips"),
    ("DEF-spine.002", "Chest",      "Spine"),
    ("DEF-spine.003", "UpperChest", "Chest"),
    ("DEF-spine.004", "Neck",       "UpperChest"),
    ("DEF-spine.005", "Head",       "Neck"),
    ("DEF-spine.006", "HeadEnd",    "Head"),
]

UNITY_LIMB_RENAMES = {
    "DEF-shoulder.L":      "LeftShoulder",
    "DEF-shoulder.R":      "RightShoulder",
    "DEF-upper_arm.L":     "LeftArm",
    "DEF-upper_arm.R":     "RightArm",
    "DEF-upper_arm.L.001": "LeftArm_twist",
    "DEF-upper_arm.R.001": "RightArm_twist",
    "DEF-forearm.L":       "LeftForeArm",
    "DEF-forearm.R":       "RightForeArm",
    "DEF-forearm.L.001":   "LeftForeArm_twist",
    "DEF-forearm.R.001":   "RightForeArm_twist",
    "DEF-hand.L":          "LeftHand",
    "DEF-hand.R":          "RightHand",
    "DEF-thigh.L":         "LeftUpLeg",
    "DEF-thigh.R":         "RightUpLeg",
    "DEF-thigh.L.001":     "LeftUpLeg_twist",
    "DEF-thigh.R.001":     "RightUpLeg_twist",
    "DEF-shin.L":          "LeftLeg",
    "DEF-shin.R":          "RightLeg",
    "DEF-shin.L.001":      "LeftLeg_twist",
    "DEF-shin.R.001":      "RightLeg_twist",
    "DEF-foot.L":          "LeftFoot",
    "DEF-foot.R":          "RightFoot",
    "DEF-toe.L":           "LeftToeBase",
    "DEF-toe.R":           "RightToeBase",
}


def rename_and_reparent_for_unity(rig):
    """
    1. Prefix any non-DEF control bones whose names collide with Unity names.
    2. Rename all DEF bones to Unity Humanoid standard names.
    3. Reparent the spine DEF chain so each bone's parent is the previous DEF
       bone — not the ORG bone Rigify uses internally.
       Without this, Unity sees 'Head' parented to 'ORG-spine.003' and rejects it.
    """
    flush("Renaming and reparenting deform skeleton for Unity Humanoid...")

    all_unity_names_lower = (
        {entry[1].lower() for entry in UNITY_SPINE_CHAIN} |
        {v.lower() for v in UNITY_LIMB_RENAMES.values()}
    )

    # Step A: prefix colliding control bones
    for bone in rig.data.bones:
        if bone.name.lower() in all_unity_names_lower and not bone.name.startswith("DEF-"):
            bone.name = "ctrl_" + bone.name
            flush(f"    Prefixed control bone → '{bone.name}'")

    # Step B: rename limb DEF bones
    for old_name, new_name in UNITY_LIMB_RENAMES.items():
        if old_name in rig.data.bones:
            rig.data.bones[old_name].name = new_name
            flush(f"    Renamed '{old_name}' → '{new_name}'")

    # Step C: rename spine DEF bones (root first)
    for old_name, new_name, _ in UNITY_SPINE_CHAIN:
        if old_name in rig.data.bones:
            rig.data.bones[old_name].name = new_name
            flush(f"    Renamed '{old_name}' → '{new_name}'")

    # Step D: reparent spine DEF chain in Edit Mode
    flush("  Reparenting DEF chain (fixing ORG parent issue for Unity)...")
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    eb = rig.data.edit_bones

    for _, bone_name, parent_name in UNITY_SPINE_CHAIN:
        if bone_name not in eb:
            eprint(f"    [skip] '{bone_name}' not in edit bones")
            continue
        if parent_name is None:
            eb[bone_name].parent = None
            flush(f"    '{bone_name}' → root")
        elif parent_name in eb:
            eb[bone_name].parent = eb[parent_name]
            eb[bone_name].use_connect = False
            flush(f"    '{bone_name}' parent → '{parent_name}'")
        else:
            eprint(f"    [warn] parent '{parent_name}' not found for '{bone_name}'")

    # Reparent limb roots to their correct Unity parents
    EXTRA_REPARENTS = [
        ("LeftShoulder",  "UpperChest"),
        ("RightShoulder", "UpperChest"),
        ("LeftUpLeg",     "Hips"),
        ("RightUpLeg",    "Hips"),
        ("LeftArm",       "LeftShoulder"),
        ("RightArm",      "RightShoulder"),
        ("LeftForeArm",   "LeftArm"),
        ("RightForeArm",  "RightArm"),
        ("LeftHand",      "LeftForeArm"),
        ("RightHand",     "RightForeArm"),
        ("LeftLeg",       "LeftUpLeg"),
        ("RightLeg",      "RightUpLeg"),
        ("LeftFoot",      "LeftLeg"),
        ("RightFoot",     "RightLeg"),
        ("LeftToeBase",   "LeftFoot"),
        ("RightToeBase",  "RightFoot"),
    ]
    for bone_name, parent_name in EXTRA_REPARENTS:
        if bone_name in eb and parent_name in eb:
            eb[bone_name].parent = eb[parent_name]
            eb[bone_name].use_connect = False
            flush(f"    '{bone_name}' parent → '{parent_name}'")

    bpy.ops.object.mode_set(mode='OBJECT')
    flush("  Unity reparenting complete.")


# ─────────────────────────────────────────────────────────────────────────────
#  Fix toe vertex weights
# ─────────────────────────────────────────────────────────────────────────────

def fix_toe_weights(body_obj, joints):
    """
    Automatic weights often assign toe vertices to the foot/shin bone.
    This finds vertices below and forward of the foot joint and reassigns
    them exclusively to LeftToeBase / RightToeBase vertex groups.
    """
    def jv(name):
        return mathutils.Vector(joints[name]) if name in joints else None

    l_foot_pos = jv("left_foot")
    r_foot_pos = jv("right_foot")
    l_ankle    = jv("left_ankle")
    r_ankle    = jv("right_ankle")

    if not l_foot_pos or not r_foot_pos:
        flush("  [toe fix] foot joints missing, skipping.")
        return

    mw = body_obj.matrix_world

    # Find left and right toe vertex group names (renamed to Unity names)
    def find_vg(keywords):
        for vg in body_obj.vertex_groups:
            for kw in keywords:
                if kw.lower() in vg.name.lower():
                    return vg
        return None

    l_toe_vg = find_vg(["LeftToeBase", "DEF-toe.L", "toe.L"])
    r_toe_vg = find_vg(["RightToeBase", "DEF-toe.R", "toe.R"])
    l_foot_vg = find_vg(["LeftFoot", "DEF-foot.L", "foot.L"])
    r_foot_vg = find_vg(["RightFoot", "DEF-foot.R", "foot.R"])

    flush(f"  [toe fix] L toe vg: {l_toe_vg.name if l_toe_vg else None}")
    flush(f"  [toe fix] R toe vg: {r_toe_vg.name if r_toe_vg else None}")

    if not l_toe_vg or not r_toe_vg:
        flush("  [toe fix] Toe vertex groups not found, skipping.")
        return

    # Vertices are "toe verts" if they are:
    #   - Forward of the foot joint (-Y direction, i.e. Y < foot_Y)
    #   - Within a lateral band around the foot (X within foot_X ± 0.08)
    #   - At roughly foot height or below (Z <= foot_Z + 0.03)
    l_toe_verts = []
    r_toe_verts = []

    for v in body_obj.data.vertices:
        wp = mw @ v.co
        # Left toe region
        if (wp.y < l_foot_pos.y and
            abs(wp.x - l_foot_pos.x) < 0.09 and
            wp.z <= l_foot_pos.z + 0.03):
            l_toe_verts.append(v.index)
        # Right toe region
        if (wp.y < r_foot_pos.y and
            abs(wp.x - r_foot_pos.x) < 0.09 and
            wp.z <= r_foot_pos.z + 0.03):
            r_toe_verts.append(v.index)

    flush(f"  [toe fix] L toe verts: {len(l_toe_verts)}, R toe verts: {len(r_toe_verts)}")

    # For left toes: remove from foot/shin, assign to toe bone
    other_vgs = [vg for vg in body_obj.vertex_groups
                 if vg.name != l_toe_vg.name]
    for vg in other_vgs:
        try: vg.remove(l_toe_verts)
        except: pass
    l_toe_vg.add(l_toe_verts, 1.0, 'REPLACE')

    # For right toes: remove from foot/shin, assign to toe bone
    other_vgs = [vg for vg in body_obj.vertex_groups
                 if vg.name != r_toe_vg.name]
    for vg in other_vgs:
        try: vg.remove(r_toe_verts)
        except: pass
    r_toe_vg.add(r_toe_verts, 1.0, 'REPLACE')

    flush(f"  [toe fix] Toe weights reassigned.")

# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    flush(f"Blender version : {bpy.app.version_string}")
    flush(f"Args received   : {args}")

    if len(args) < 2:
        eprint("Usage: blender --background --python rig_smplx_with_rigify.py -- joints.json mesh.obj [out.blend]")
        return

    joints_path = os.path.abspath(args[0])
    obj_path    = os.path.abspath(args[1])
    file_root = os.path.splitext(os.path.basename(obj_path))[0]
    out_blend = os.path.abspath(os.path.join("outputs", f"{file_root}_rigged.blend"))    
    if not os.path.exists(joints_path):
        eprint("Parameter file not found:", joints_path)
        return

    flush(f"Joints JSON  : {joints_path}")
    flush(f"OBJ mesh     : {obj_path}")
    flush(f"Output .blend: {out_blend}")

    if not os.path.exists(joints_path):
        eprint("ERROR: joints JSON not found:", joints_path); return
    if not os.path.exists(obj_path):
        eprint("ERROR: OBJ mesh not found:", obj_path); return

    if not enable_rigify():
        eprint("Aborting: Rigify could not be enabled."); return

    flush("Clearing default scene...")
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    flush("Importing OBJ...")
    if bpy.app.version >= (3, 2, 0):
        bpy.ops.wm.obj_import(filepath=obj_path)
    else:
        bpy.ops.import_scene.obj(filepath=obj_path)

    mesh_objs = [o for o in bpy.context.selected_objects if o.type == 'MESH']
    if not mesh_objs:
        eprint("ERROR: No mesh found after OBJ import."); return
    mesh_obj = mesh_objs[0]
    mesh_obj.name = "SMPLX_Body"
    flush(f"  Mesh imported: {mesh_obj.name} ({len(mesh_obj.data.vertices)} verts)")

    with open(joints_path) as f:
        joints = json.load(f)
    flush(f"  Loaded {len(joints)} joints.")

    flush("Adding Human MetaRig...")
    bpy.ops.object.select_all(action='DESELECT')
    try:
        bpy.ops.object.armature_human_metarig_add()
    except Exception as e:
        eprint(f"ERROR: armature_human_metarig_add failed: {e}")
        traceback.print_exc(); return

    metarig = bpy.context.object
    if metarig is None or metarig.type != 'ARMATURE':
        eprint("ERROR: MetaRig was not created."); return
    metarig.name = "metarig"
    flush(f"  MetaRig created: {metarig.name}")

    flush("Positioning bones from SMPL-X joints...")
    bpy.ops.object.mode_set(mode='EDIT')
    strip_face_and_finger_bones(metarig)
    position_metarig_bones(metarig, joints)
    bpy.ops.object.mode_set(mode='OBJECT')

    flush("Generating Rigify rig...")
    bpy.context.view_layer.objects.active = metarig
    metarig.select_set(True)
    bpy.ops.object.mode_set(mode='POSE')
    try:
        bpy.ops.pose.rigify_generate()
    except Exception as e:
        eprint(f"ERROR: Rigify generation failed: {e}")
        traceback.print_exc()
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.wm.save_as_mainfile(filepath=out_blend)
        flush(f"Partial file saved: {out_blend}")
        return

    bpy.ops.object.mode_set(mode='OBJECT')
    rig = bpy.data.objects.get("rig") or bpy.context.object
    rig.name = "RigifyRig"
    metarig.hide_viewport = True
    flush(f"  Rigify rig generated: {rig.name}")

    # ── Rename + reparent DEF bones for Unity ─────────────────────
    rename_and_reparent_for_unity(rig)

    # ── Separate mesh islands and skin each to its bone ───────────
    body_obj = separate_and_skin_islands(mesh_obj, rig, joints)

    # ── Skin main body with automatic weights ─────────────────────
    flush("Binding body mesh to rig with automatic weights...")
    bpy.ops.object.select_all(action='DESELECT')
    body_obj.select_set(True)
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.parent_set(type='ARMATURE_AUTO')
    flush("  Body skinning done.")

    flush("Fixing breast vertex weights...")
    fix_breast_weights(body_obj, rig, joints)

    flush("Fixing toe vertex weights...")
    fix_toe_weights(body_obj, joints)

    bpy.ops.wm.save_as_mainfile(filepath=out_blend)
    flush(f"\nSuccess! Saved: {out_blend}")


if __name__ == '__main__':
    try:
        main()
    except Exception:
        eprint("UNCAUGHT EXCEPTION:")
        traceback.print_exc()
        sys.exit(1)