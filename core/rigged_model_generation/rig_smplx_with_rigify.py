#!/usr/bin/env python3
"""
Rig an SMPL-X OBJ mesh using a direct Blender armature (no Rigify).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY NO RIGIFY:
  - Rigify generates IK/FK control layers that corrupt the DEF-bone transforms.
  - Rigify's metarig has its own internal Y-axis orientation; mapping SMPL-X
    joints into it always introduces tilt or scale artefacts.
  - Auto-weights on a Rigify rig target ORG-bones instead of DEF-bones,
    causing the "stretching / morphing" artefact.

THIS SCRIPT:
  1. Imports the OBJ mesh.
  2. Reads joints.json (already in Blender Z-up space — no axis swap needed).
  3. Builds a single armature whose bones map 1-to-1 to the SMPL-X skeleton.
  4. Names every bone to the Unity Humanoid standard for direct Unity import.
  5. Binds the mesh with Blender's heat-map automatic weights.
  6. Saves a .blend and exports a .fbx (Unity-ready).

Usage (headless Blender):
  blender --background --python rig_smplx_direct.py -- joints.json mesh.obj [out_dir]

Requirements:
  Blender 3.x or 4.x  (no addons needed)
"""

import bpy
import sys
import os
import json
import mathutils
import traceback


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def flush(*a, **kw):
    print(*a, **kw); sys.stdout.flush()

def eprint(*a, **kw):
    print(*a, **kw, file=sys.stderr); sys.stderr.flush()

def parse_args():
    argv = sys.argv
    return argv[argv.index("--") + 1:] if "--" in argv else []

def vec(joints, name):
    """Return a mathutils.Vector from the joints dict, or None."""
    v = joints.get(name)
    return mathutils.Vector(v) if v is not None else None


# ─────────────────────────────────────────────────────────────────────────────
#  Skeleton definition
#
#  Each entry: (unity_bone_name, joint_head, joint_tail, parent_unity_name)
#
#  joint_head  = key in joints.json that becomes bone.head
#  joint_tail  = key in joints.json that becomes bone.tail
#                (if None the tail is computed as head + small offset)
#  parent      = unity name of parent bone (None → root)
#
#  NOTE: joints.json values are already in Blender's Z-up coordinate space.
#        The SMPL-X exporter writes X=right, Y=forward, Z=up which matches
#        Blender exactly — so NO axis swap is applied here.
# ─────────────────────────────────────────────────────────────────────────────

SKELETON = [
    # ── Spine ──────────────────────────────────────────────────────────────
    ("Hips",        "pelvis",   "spine1",         None),
    ("Spine",       "spine1",   "spine2",          "Hips"),
    ("Chest",       "spine2",   "spine3",          "Spine"),
    ("UpperChest",  "spine3",   "neck",            "Chest"),
    ("Neck",        "neck",     "head",            "UpperChest"),
    ("Head",        "head",     None,              "Neck"),       # tail computed

    # ── Left leg ───────────────────────────────────────────────────────────
    ("LeftUpLeg",   "left_hip",    "left_knee",   "Hips"),
    ("LeftLeg",     "left_knee",   "left_ankle",  "LeftUpLeg"),
    ("LeftFoot",    "left_ankle",  "left_foot",   "LeftLeg"),
    ("LeftToeBase", "left_foot",   None,          "LeftFoot"),    # tail computed

    # ── Right leg ──────────────────────────────────────────────────────────
    ("RightUpLeg",  "right_hip",   "right_knee",  "Hips"),
    ("RightLeg",    "right_knee",  "right_ankle", "RightUpLeg"),
    ("RightFoot",   "right_ankle", "right_foot",  "RightLeg"),
    ("RightToeBase","right_foot",  None,          "RightFoot"),   # tail computed

    # ── Left arm ───────────────────────────────────────────────────────────
    ("LeftShoulder","left_collar",   "left_shoulder",  "UpperChest"),
    ("LeftArm",     "left_shoulder", "left_elbow",     "LeftShoulder"),
    ("LeftForeArm", "left_elbow",    "left_wrist",     "LeftArm"),
    ("LeftHand",    "left_wrist",    None,             "LeftForeArm"),  # tail computed

    # ── Right arm ──────────────────────────────────────────────────────────
    ("RightShoulder","right_collar",  "right_shoulder", "UpperChest"),
    ("RightArm",     "right_shoulder","right_elbow",    "RightShoulder"),
    ("RightForeArm", "right_elbow",   "right_wrist",    "RightArm"),
    ("RightHand",    "right_wrist",   None,             "RightForeArm"), # tail computed

    # ── Left hand fingers ──────────────────────────────────────────────────
    ("LeftThumb1",      "left_thumb1",  "left_thumb2",  "LeftHand"),
    ("LeftThumb2",      "left_thumb2",  "left_thumb3",  "LeftThumb1"),
    ("LeftThumb3",      "left_thumb3",  None,           "LeftThumb2"),
    ("LeftIndex1",      "left_index1",  "left_index2",  "LeftHand"),
    ("LeftIndex2",      "left_index2",  "left_index3",  "LeftIndex1"),
    ("LeftIndex3",      "left_index3",  None,           "LeftIndex2"),
    ("LeftMiddle1",     "left_middle1", "left_middle2", "LeftHand"),
    ("LeftMiddle2",     "left_middle2", "left_middle3", "LeftMiddle1"),
    ("LeftMiddle3",     "left_middle3", None,           "LeftMiddle2"),
    ("LeftRing1",       "left_ring1",   "left_ring2",   "LeftHand"),
    ("LeftRing2",       "left_ring2",   "left_ring3",   "LeftRing1"),
    ("LeftRing3",       "left_ring3",   None,           "LeftRing2"),
    ("LeftPinky1",      "left_pinky1",  "left_pinky2",  "LeftHand"),
    ("LeftPinky2",      "left_pinky2",  "left_pinky3",  "LeftPinky1"),
    ("LeftPinky3",      "left_pinky3",  None,           "LeftPinky2"),

    # ── Right hand fingers ─────────────────────────────────────────────────
    ("RightThumb1",     "right_thumb1",  "right_thumb2",  "RightHand"),
    ("RightThumb2",     "right_thumb2",  "right_thumb3",  "RightThumb1"),
    ("RightThumb3",     "right_thumb3",  None,            "RightThumb2"),
    ("RightIndex1",     "right_index1",  "right_index2",  "RightHand"),
    ("RightIndex2",     "right_index2",  "right_index3",  "RightIndex1"),
    ("RightIndex3",     "right_index3",  None,            "RightIndex2"),
    ("RightMiddle1",    "right_middle1", "right_middle2", "RightHand"),
    ("RightMiddle2",    "right_middle2", "right_middle3", "RightMiddle1"),
    ("RightMiddle3",    "right_middle3", None,            "RightMiddle2"),
    ("RightRing1",      "right_ring1",   "right_ring2",   "RightHand"),
    ("RightRing2",      "right_ring2",   "right_ring3",   "RightRing1"),
    ("RightRing3",      "right_ring3",   None,            "RightRing2"),
    ("RightPinky1",     "right_pinky1",  "right_pinky2",  "RightHand"),
    ("RightPinky2",     "right_pinky2",  "right_pinky3",  "RightPinky1"),
    ("RightPinky3",     "right_pinky3",  None,            "RightPinky2"),
]

# Minimum bone length — prevents zero-length bones that crash Blender
MIN_BONE_LEN = 0.01


def _auto_tail(head_vec, parent_head_vec, fallback_dir=None):
    """
    For end-effector bones (no explicit tail joint):
    extend 40% of the parent bone's length in the same direction,
    with a sensible minimum.
    """
    if parent_head_vec is not None:
        direction = head_vec - parent_head_vec
        length = max(direction.length * 0.4, MIN_BONE_LEN)
        if direction.length > 1e-5:
            return head_vec + direction.normalized() * length

    if fallback_dir:
        return head_vec + mathutils.Vector(fallback_dir) * MIN_BONE_LEN * 3

    return head_vec + mathutils.Vector((0, 0, MIN_BONE_LEN * 3))


# ─────────────────────────────────────────────────────────────────────────────
#  Build armature
# ─────────────────────────────────────────────────────────────────────────────

def build_armature(joints):
    """
    Create a Blender armature whose bones directly mirror the SMPL-X joints.
    Returns the armature object.
    """
    flush("Building direct armature from SMPL-X joints...")

    arm_data = bpy.data.armatures.new("SMPLX_Armature")
    arm_obj  = bpy.data.objects.new("SMPLX_Rig", arm_data)
    bpy.context.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj
    arm_obj.select_set(True)

    bpy.ops.object.mode_set(mode='EDIT')
    eb = arm_data.edit_bones

    # First pass: create all bones with correct head/tail
    bone_heads = {}   # unity_name → Vector (for auto-tail computation)

    for (bone_name, head_key, tail_key, parent_name) in SKELETON:
        head_vec = vec(joints, head_key)
        if head_vec is None:
            flush(f"  [skip] '{bone_name}': joint '{head_key}' not in JSON")
            continue

        bone_heads[bone_name] = head_vec

        if tail_key is not None:
            tail_vec = vec(joints, tail_key)
            if tail_vec is None:
                tail_vec = _auto_tail(head_vec, None)
        else:
            # Compute tail from parent bone's direction
            parent_head = bone_heads.get(parent_name)
            tail_vec = _auto_tail(head_vec, parent_head)

        # Enforce minimum length
        if (tail_vec - head_vec).length < MIN_BONE_LEN:
            tail_vec = head_vec + mathutils.Vector((0, 0, MIN_BONE_LEN))

        b = eb.new(bone_name)
        b.head = head_vec
        b.tail = tail_vec

    # Second pass: set parents
    for (bone_name, _, _, parent_name) in SKELETON:
        if bone_name not in eb:
            continue
        if parent_name and parent_name in eb:
            eb[bone_name].parent = eb[parent_name]
            eb[bone_name].use_connect = False   # keep free transforms

    bpy.ops.object.mode_set(mode='OBJECT')

    bone_count = len([b for b in arm_data.bones])
    flush(f"  Armature built: {bone_count} bones")
    return arm_obj


# ─────────────────────────────────────────────────────────────────────────────
#  Bind mesh → armature with automatic weights
# ─────────────────────────────────────────────────────────────────────────────

def bind_mesh(mesh_obj, rig_obj):
    """
    Parent mesh to armature with heat-map automatic weights.
    This is Blender's most accurate built-in skinning method and works
    correctly because we have a 1-to-1 direct armature (no Rigify layers).
    """
    flush("Binding mesh to armature (automatic weights)...")

    bpy.ops.object.select_all(action='DESELECT')
    mesh_obj.select_set(True)
    rig_obj.select_set(True)
    bpy.context.view_layer.objects.active = rig_obj

    bpy.ops.object.parent_set(type='ARMATURE_AUTO')
    flush("  Automatic weight binding complete.")


# ─────────────────────────────────────────────────────────────────────────────
#  Post-process: clamp extreme weights to avoid stretching
# ─────────────────────────────────────────────────────────────────────────────

def clamp_weights(mesh_obj, max_weight=0.85):
    """
    Cap any vertex weight above max_weight.
    Prevents a single bone from pulling a vertex >85% — the main cause of
    the "stretching / morphing" artefact in hands and legs.
    """
    flush(f"  Clamping extreme vertex weights (cap={max_weight})...")
    clamped = 0
    for v in mesh_obj.data.vertices:
        total = sum(g.weight for g in v.groups)
        if total < 1e-6:
            continue
        for g in v.groups:
            if g.weight > max_weight:
                g.weight = max_weight
                clamped += 1
    flush(f"  Clamped {clamped} weight values.")


# ─────────────────────────────────────────────────────────────────────────────
#  Separate mesh islands and skin small parts 100% to nearest bone
# ─────────────────────────────────────────────────────────────────────────────

def separate_islands(mesh_obj):
    """Split loose mesh islands into separate objects. Returns list of all mesh objects."""
    bpy.ops.object.select_all(action='DESELECT')
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.separate(type='LOOSE')
    bpy.ops.object.mode_set(mode='OBJECT')

    islands = [o for o in bpy.context.scene.objects
               if o.type == 'MESH' and not o.name.startswith("WGT-")]
    islands.sort(key=lambda o: len(o.data.vertices), reverse=True)
    flush(f"  Separated into {len(islands)} mesh islands.")
    return islands


def skin_small_island_to_nearest_bone(island, rig, joints):
    """
    For small islands (eyes, teeth, etc.) assign all vertices 100%
    to the nearest bone head in joint space.
    """
    center = sum(
        (island.matrix_world @ mathutils.Vector(c) for c in island.bound_box),
        mathutils.Vector()
    ) / 8.0

    best_bone = None
    best_dist = float('inf')
    for bone_name, head_key, _, _ in SKELETON:
        jv = vec(joints, head_key)
        if jv is None:
            continue
        d = (center - jv).length
        if d < best_dist:
            best_dist = d
            best_bone = bone_name

    if best_bone is None:
        flush(f"    [warn] Could not find nearest bone for '{island.name}'")
        return

    # Parent to rig
    mat = island.matrix_world.copy()
    island.parent = rig
    island.parent_type = 'OBJECT'
    island.matrix_world = mat

    if not any(m.type == 'ARMATURE' for m in island.modifiers):
        mod = island.modifiers.new("Armature", 'ARMATURE')
        mod.object = rig

    vg = island.vertex_groups.get(best_bone) or island.vertex_groups.new(name=best_bone)
    vg.add([v.index for v in island.data.vertices], 1.0, 'REPLACE')
    flush(f"    '{island.name}' ({len(island.data.vertices)}v) → bone '{best_bone}' (d={best_dist:.3f}m)")


# ─────────────────────────────────────────────────────────────────────────────
#  Export FBX (Unity-ready)
# ─────────────────────────────────────────────────────────────────────────────

def export_fbx(filepath):
    flush(f"Exporting FBX → {filepath}")
    try:
        bpy.ops.export_scene.fbx(
            filepath=filepath,
            use_selection=False,
            apply_unit_scale=True,
            apply_scale_options='FBX_SCALE_ALL',
            bake_space_transform=False,
            object_types={'ARMATURE', 'MESH'},
            use_mesh_modifiers=True,
            add_leaf_bones=False,
            primary_bone_axis='Y',
            secondary_bone_axis='X',
            armature_nodetype='NULL',
            bake_anim=False,
            path_mode='AUTO',
        )
        flush("  FBX exported.")
    except Exception as e:
        eprint(f"  [warn] FBX export failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    flush(f"Blender {bpy.app.version_string}")
    flush(f"Args: {args}")

    if len(args) < 2:
        eprint("Usage: blender --background --python rig_smplx_direct.py -- joints.json mesh.obj [out_dir]")
        sys.exit(1)

    joints_path = os.path.abspath(args[0])
    obj_path    = os.path.abspath(args[1])
    out_dir     = os.path.abspath(args[2]) if len(args) > 2 else os.path.abspath("outputs")

    os.makedirs(out_dir, exist_ok=True)

    stem       = os.path.splitext(os.path.basename(obj_path))[0]
    out_blend  = os.path.join(out_dir, f"{stem}_rigged.blend")
    out_fbx    = os.path.join(out_dir, f"{stem}_rigged.fbx")

    flush(f"Joints JSON : {joints_path}")
    flush(f"OBJ mesh    : {obj_path}")
    flush(f"Output dir  : {out_dir}")

    for p in (joints_path, obj_path):
        if not os.path.exists(p):
            eprint(f"ERROR: file not found: {p}"); sys.exit(1)

    # ── Load joints ───────────────────────────────────────────────────────
    with open(joints_path) as f:
        joints = json.load(f)
    flush(f"Loaded {len(joints)} joints.")

    # ── Clear scene ───────────────────────────────────────────────────────
    flush("Clearing scene...")
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # ── Import OBJ ────────────────────────────────────────────────────────
    flush("Importing OBJ...")
    if bpy.app.version >= (3, 2, 0):
        bpy.ops.wm.obj_import(filepath=obj_path)
    else:
        bpy.ops.import_scene.obj(filepath=obj_path)

    mesh_objs = [o for o in bpy.context.selected_objects if o.type == 'MESH']
    if not mesh_objs:
        eprint("ERROR: No mesh after import."); sys.exit(1)

    mesh_obj = mesh_objs[0]
    mesh_obj.name = "SMPLX_Body"
    flush(f"Mesh: {mesh_obj.name} ({len(mesh_obj.data.vertices)} verts)")

    # ── Check mesh orientation ────────────────────────────────────────────
    # SMPL-X OBJ meshes are exported Z-up by default.
    # If your OBJ was exported Y-up, uncomment the next two lines:
    # mesh_obj.rotation_euler = (0, 0, 0)
    # bpy.ops.object.transform_apply(rotation=True)
    flush(f"Mesh bounding box Z: {min(v.co.z for v in mesh_obj.data.vertices):.3f} → "
          f"{max(v.co.z for v in mesh_obj.data.vertices):.3f}")

    # ── Separate mesh islands ─────────────────────────────────────────────
    islands = separate_islands(mesh_obj)

    # Largest island = body
    body_obj = islands[0]
    body_obj.name = "SMPLX_Body"

    # ── Build armature ────────────────────────────────────────────────────
    rig = build_armature(joints)

    # ── Bind body mesh with automatic weights ─────────────────────────────
    bind_mesh(body_obj, rig)

    # ── Clamp extreme weights to prevent stretching ───────────────────────
    clamp_weights(body_obj, max_weight=0.85)

    # ── Skin small islands (eyes, teeth, etc.) to nearest bone ───────────
    SMALL_ISLAND_THRESHOLD = 2000   # verts; anything below this is an accessory
    flush("Skinning small mesh islands...")
    for island in islands[1:]:
        flush(f"  Processing: {island.name} ({len(island.data.vertices)}v)")
        if len(island.data.vertices) < SMALL_ISLAND_THRESHOLD:
            skin_small_island_to_nearest_bone(island, rig, joints)
        else:
            # Large secondary island — also auto-weight
            bind_mesh(island, rig)
            clamp_weights(island)

    # ── Save .blend ───────────────────────────────────────────────────────
    bpy.ops.wm.save_as_mainfile(filepath=out_blend)
    flush(f"Saved .blend → {out_blend}")

    # ── Export FBX ────────────────────────────────────────────────────────
    export_fbx(out_fbx)

    flush(f"\n✓ Done.  Output files:")
    flush(f"  {out_blend}")
    flush(f"  {out_fbx}")


if __name__ == '__main__':
    try:
        main()
    except Exception:
        eprint("UNCAUGHT EXCEPTION:")
        traceback.print_exc()
        sys.exit(1)