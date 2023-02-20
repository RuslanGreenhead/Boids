# Boids
Model to simulate birds flocking (["Boids"](https://en.wikipedia.org/wiki/Boids), by Craig Reynolds, 1986)\
In a shortcut: we generate random points on a plane with random velocities and then modify their accelerations according to the following steers:
- **Cohesion**\
![Image](https://upload.wikimedia.org/wikipedia/commons/2/2b/Rule_cohesion.gif)
- **Alignment**\
![Image](https://upload.wikimedia.org/wikipedia/commons/e/e1/Rule_alignment.gif)
- **Separation**\
![Image](https://upload.wikimedia.org/wikipedia/commons/e/e1/Rule_separation.gif)
