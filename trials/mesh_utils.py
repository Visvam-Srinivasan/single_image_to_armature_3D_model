import torch
import smplx
import pyvista as pv
import numpy as np
import os

class MeshGenerator:
    def __init__(self, model_path='models/', model_type='smplx', gender='male'):
        """Initializes the SMPL-X model and tracking for the interactive UI."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Ensure path exists for Linux environments
        if not os.path.exists(model_path):
            print(f"Warning: model_path '{model_path}' not found. Check your directory.")

        self.model = smplx.create(
            model_path=model_path, 
            model_type=model_type, 
            gender=gender
        ).to(self.device)
        
        self.faces = self.model.faces
        self.sliders = {}
        self.plotter = None
        self.actor = None

    def create_mesh_data(self, betas=None, expression=None):
        """Generates mesh data for either export or visualization."""
        if betas is None: betas = torch.zeros([1, 10])
        if expression is None: expression = torch.zeros([1, 10])

        output = self.model(
            betas=betas.to(self.device), 
            expression=expression.to(self.device), 
            return_verts=True
        )
        verts = output.vertices.detach().cpu().numpy().squeeze()
        
        # PyVista face format: [3, v1, v2, v3, ...]
        pv_faces = np.column_stack((np.full(len(self.faces), 3), self.faces))
        return pv.PolyData(verts, pv_faces)

    def _update_scene(self, value):
        """Callback for sliders to update the 3D model in real-time."""
        betas = torch.zeros([1, 10])
        for i in range(10):
            if f'beta_{i}' in self.sliders:
                betas[0, i] = self.sliders[f'beta_{i}'].GetRepresentation().GetValue()

        expression = torch.zeros([1, 10])
        for i in range(10):
            if f'exp_{i}' in self.sliders:
                expression[0, i] = self.sliders[f'exp_{i}'].GetRepresentation().GetValue()

        new_mesh = self.create_mesh_data(betas, expression)
        if self.actor:
            self.actor.mapper.dataset.points = new_mesh.points

    def launch_interactive(self):
        """Runs the interactive PyVista window with white, non-overlapping slider text."""
        self.plotter = pv.Plotter(shape=(1, 2), window_size=[1400, 850])
        
        # Right Viewport: The Model
        self.plotter.subplot(0, 1)
        initial_mesh = self.create_mesh_data()
        self.actor = self.plotter.add_mesh(initial_mesh, color='tan', smooth_shading=True)
        self.plotter.camera_position = 'xy'
        self.plotter.reset_camera()

        # Left Viewport: The UI
        self.plotter.subplot(0, 0)
        self.plotter.set_background('#1e1e26')

        for i in range(10):
            # Reduced vertical spacing slightly to ensure titles don't touch next slider
            y_pos = 0.94 - (i * 0.09) 
            
            # --- Beta Sliders ---
            slider_beta = self.plotter.add_slider_widget(
                callback=self._update_scene, rng=[-5, 5], value=0, title=f'Shape {i}',
                pointa=(0.05, y_pos), pointb=(0.40, y_pos), style='modern'
            )
            
            # Formatting to prevent overlap
            rep_b = slider_beta.GetRepresentation()
            rep_b.GetTitleProperty().SetColor(1, 1, 1)
            rep_b.GetLabelProperty().SetColor(1, 1, 1)
            # Scaling down heights fixes the overlap
            rep_b.SetTitleHeight(0.015) 
            rep_b.SetLabelHeight(0.015)
            # This pushes the title further above the bar
            rep_b.GetTitleProperty().SetVerticalJustificationToTop() 
            
            self.sliders[f'beta_{i}'] = slider_beta

            # --- Exp Sliders ---
            slider_exp = self.plotter.add_slider_widget(
                callback=self._update_scene, rng=[-3, 3], value=0, title=f'Exp {i}',
                pointa=(0.55, y_pos), pointb=(0.90, y_pos), style='modern'
            )
            
            # Formatting to prevent overlap
            rep_e = slider_exp.GetRepresentation()
            rep_e.GetTitleProperty().SetColor(1, 1, 1)
            rep_e.GetLabelProperty().SetColor(1, 1, 1)
            rep_e.SetTitleHeight(0.015)
            rep_e.SetLabelHeight(0.015)
            rep_e.GetTitleProperty().SetVerticalJustificationToTop()

            self.sliders[f'exp_{i}'] = slider_exp

        print("Launching Interactive SMPL-X Viewer...")
        self.plotter.show()

# --- STANDALONE EXECUTION ---
if __name__ == "__main__":
    # Ensure this runs correctly from the 3D_reconstruction root folder
    gen = MeshGenerator(model_path='models/')
    gen.launch_interactive()