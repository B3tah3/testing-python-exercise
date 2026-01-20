"""
Tests for functionality checks in class SolveDiffusion2D
"""

from diffusion2d import SolveDiffusion2D
import numpy as np

def test_initialize_physical_parameters():
    """
    Checks function SolveDiffusion2D.initialize_domain
    """
    solver = SolveDiffusion2D()
    solver.initialize_domain(w=10., h=10., dx=0.1, dy=0.1)
    solver.initialize_physical_parameters(d=5., T_cold=200., T_hot=800.)
    #test this: dx2, dy2 = self.dx * self.dx, self.dy * self.dy
    #    self.dt = dx2 * dy2 / (2 * self.D * (dx2 + dy2))
    # expected value: dx2 = dy2 = 0.01
    # dt = 0.0001 / (2 * 5 * 0.02) = 0.0001 / 0.2 = 0.0005
    assert((solver.dt - 0.0005) < 0.00000001)

def test_set_initial_condition():
    """
    Checks function SolveDiffusion2D.get_initial_function
    """
    solver = SolveDiffusion2D()
    solver.initialize_domain(w=10., h=10., dx=0.1, dy=0.1)
    solver.initialize_physical_parameters(d=5., T_cold=200., T_hot=800.)
    u = solver.set_initial_condition()

    referenceU = solver.T_cold * np.ones((solver.nx, solver.ny))

    # Initial conditions - circle of radius r centred at (cx,cy) (mm)
    r, cx, cy = 2, 5, 5
    r2 = r ** 2
    for i in range(solver.nx):
        for j in range(solver.ny):
            p2 = (i * solver.dx - cx) ** 2 + (j * solver.dy - cy) ** 2
            if p2 < r2:
                referenceU[i, j] = solver.T_hot

    np.testing.assert_array_equal(u, referenceU)
