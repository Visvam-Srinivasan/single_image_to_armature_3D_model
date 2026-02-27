import torch
import smplx
import pyvista as pv
import numpy as np
import os
import math
import time

class AnimatedMeshGenerator:
    def __init__(self, model_path='models/', model_type='smplx', gender='male'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"model_path '{model_path}' not found")

        self.model = smplx.create(
            model_path=model_path,
            model_type=model_type,
            gender=gender
        ).to(self.device)

        self.faces = self.model.faces
        self.plotter = None
        self.actor = None
        self.t = 0.0
        self.playing = False   # animation state

    def create_mesh(self, betas, expression):
        output = self.model(
            betas=betas.to(self.device),
            expression=expression.to(self.device),
            return_verts=True
        )
        verts = output.vertices.detach().cpu().numpy().squeeze()
        pv_faces = np.column_stack((np.full(len(self.faces), 3), self.faces))
        return pv.PolyData(verts, pv_faces)

    def update(self):
        self.t += 0.05

        betas = torch.zeros([1, 10])
        expression = torch.zeros([1, 10])

        for i in range(10):
            betas[0, i] = 2.0 * math.sin(self.t + i)
            expression[0, i] = 1.5 * math.sin(self.t * 1.3 + i)

        new_mesh = self.create_mesh(betas, expression)
        self.actor.mapper.dataset.points[:] = new_mesh.points

    # ---- KEY CALLBACK ----
    def toggle_play(self):
        self.playing = not self.playing
        print("PLAYING" if self.playing else "PAUSED")

    def launch_animation(self):
        self.plotter = pv.Plotter(window_size=[900, 900])
        self.plotter.set_background("#1e1e26")

        mesh = self.create_mesh(
            torch.zeros([1, 10]),
            torch.zeros([1, 10])
        )

        self.actor = self.plotter.add_mesh(
            mesh,
            color="tan",
            smooth_shading=True
        )

        self.plotter.camera_position = "xy"
        self.plotter.reset_camera()

        # ---- KEY BINDING (SPACEBAR) ----
        self.plotter.add_key_event("space", self.toggle_play)

        print("SPACE = Play / Pause animation")
        self.plotter.show(auto_close=False)

        # ---- MAIN LOOP ----
        try:
            while True:
                if self.playing:
                    self.update()
                self.plotter.render()
                time.sleep(0.03)  # ~30 FPS
        except KeyboardInterrupt:
            pass

        self.plotter.close()


if __name__ == "__main__":
    gen = AnimatedMeshGenerator(model_path="models/")
    gen.launch_animation()
