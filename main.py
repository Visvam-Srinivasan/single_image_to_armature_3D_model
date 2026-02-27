import torch
from core.mesh_utils import MeshGenerator

def main():
    # Initialize the generator
    # It will automatically detect if you have a GPU
    gen = MeshGenerator(model_path='models/', gender='male')

    # Define your parameters (these could come from your ML models later)
    shape_params = torch.randn([1, 10])
    face_params = torch.randn([1, 10])

    # Call the generation logic
    print("Generating 3D mesh...")
    mesh, saved_path = gen.create_mesh(
        betas=shape_params, 
        expression=face_params, 
        output_filename='outputs/reconstructed_avatar.obj'
    )

    print(f"Mesh successfully saved to: {saved_path}")
    
    # Optional: View the result (note: requires GUI/X11 on Linux)
    # mesh.show()

if __name__ == "__main__":
    main()