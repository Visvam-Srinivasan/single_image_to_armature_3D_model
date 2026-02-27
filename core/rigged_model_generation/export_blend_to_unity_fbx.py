import bpy
import os
import sys

def export_blend_to_unity_fbx(blend_file_path):
    # 1. Load the blend file
    if not os.path.exists(blend_file_path):
        print(f"Error: {blend_file_path} not found.")
        return

    bpy.ops.wm.open_mainfile(filepath=blend_file_path)

    # 2. Setup paths
    # Get the directory where the .blend is located
    current_dir = os.path.dirname(blend_file_path)
    
    # Logic fix: If the blend is already in 'outputs', don't create 'outputs/outputs'
    # We will save the FBX in the same folder as the .blend
    base_name = os.path.splitext(os.path.basename(blend_file_path))[0]
    fbx_path = os.path.abspath(os.path.join(current_dir, f"{base_name}.fbx"))

    # Ensure the directory exists (just in case)
    os.makedirs(os.path.dirname(fbx_path), exist_ok=True)

    # 3. Selection Logic
    # We look for the Rigify rig (skipping the WGT- shapes)
    rig = None
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE' and not obj.name.startswith("WGT-"):
            rig = obj
            break

    if not rig:
        print("Error: No Armature found in the blend file.")
        return

    # Clear current selection to ensure only desired objects are exported
    if bpy.context.view_layer.objects.active and bpy.context.view_layer.objects.active.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
        
    bpy.ops.object.select_all(action='DESELECT')

    # Select the Rig and its associated meshes
    rig.select_set(True)
    mesh_found = False
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and not obj.name.startswith("WGT-"):
            # Check if mesh is parented to rig or influenced by its armature modifier
            is_child = (obj.parent == rig)
            has_mod = any(m.type == 'ARMATURE' and m.object == rig for m in obj.modifiers)
            if is_child or has_mod:
                obj.select_set(True)
                mesh_found = True

    if not mesh_found:
        print("Warning: No meshes found associated with the rig.")

    bpy.context.view_layer.objects.active = rig

    # 4. FBX Export with Unity Humanoid Settings
    print(f"Exporting to: {fbx_path}")
    
    # 
    try:
        bpy.ops.export_scene.fbx(
            filepath=fbx_path,
            use_selection=True,
            object_types={'ARMATURE', 'MESH'},
            apply_scale_options='FBX_SCALE_ALL',
            apply_unit_scale=True,
            axis_forward='-Z',
            axis_up='Y',
            use_mesh_modifiers=True,
            primary_bone_axis='Y',
            secondary_bone_axis='X',
            use_armature_deform_only=True, # Crucial: strips Rigify control bones
            add_leaf_bones=False,
            bake_anim=False
        )
        print(f"Export Successful: {fbx_path}")
    except Exception as e:
        print(f"Export Failed: {str(e)}")

if __name__ == "__main__":
    # Command: blender --background --python export_script.py -- outputs/sample_params_rigged.blend
    argv = sys.argv
    if "--" in argv:
        input_path = argv[argv.index("--") + 1]
        export_blend_to_unity_fbx(os.path.abspath(input_path))
    else:
        print("Error: No input .blend file provided. Usage: ... -- input.blend")