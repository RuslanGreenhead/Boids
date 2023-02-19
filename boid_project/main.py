from vispy import app, scene
from vispy.geometry import Rect
import numpy as np
from funcs import (init_boids, directions, propagate, periodic_walls, flocking, walls, wall_avoidance)

w, h = 640, 480
N = 1000
dt = 0.1
asp = w / h
perception = 0.05
vrange = (0, 0.2)
#                    c     a      s      w     n
coeffs = np.array([0.5, 0.01,  0.05,  0.006, 0.5])

boids = np.zeros((N, 6), dtype=np.float64)
init_boids(boids, asp, vrange=vrange)

canvas = scene.SceneCanvas(show=True, size=(w, h))
canvas.measure_fps()
view = canvas.central_widget.add_view()
view.camera = scene.PanZoomCamera(rect=Rect(0, 0, asp, 1))
arrows = scene.Arrow(arrows=directions(boids, dt),
                     arrow_color=(1, 1, 1, 1),
                     # width=5,
                     arrow_size=5,
                     connect='segments',
                     parent=view.scene)

t_info = scene.visuals.Text("text", pos=[80, 100], parent=canvas.scene,
                       color='white', font_size=14)

def update(event):
    """
    Update the canvas
    :param event: event
    :return: None
    """
    flocking(boids, perception, coeffs, asp, vrange)
    propagate(boids, dt, vrange)
    periodic_walls(boids, asp)
    arrows.set_data(arrows=directions(boids, dt))
    t_info.text = f"N={N}\ncoh={coeffs[0]}\nalg={coeffs[1]}\nsep={coeffs[2]}\nwalls={coeffs[3]}\nnoize={coeffs[4]}\n{canvas.fps:.2f} FPS"
    canvas.update()


if __name__ == '__main__':
    timer = app.Timer(interval=0, start=True, connect=update)
    app.run()