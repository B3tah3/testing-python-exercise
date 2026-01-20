"""
Tests for functions in class SolveDiffusion2D
"""

from diffusion2d import SolveDiffusion2D

def test_initialize_domain():
    """
    Check function SolveDiffusion2D.initialize_domain
    """
    solver = SolveDiffusion2D()
    solver.initialize_domain(w=20., h=19., dx=0.4, dy=0.4)
    #expected nx = 50 and ny = 47
    assert(solver.nx == 50)
    assert(solver.ny == 47)


def test_initialize_physical_parameters():
    """
    Checks function SolveDiffusion2D.initialize_physical_parameters
    """
    solver = SolveDiffusion2D()
    solver.dx = 0.1
    solver.dy = 0.1
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
    solver.initialize_domain()
    solver.initialize_physical_parameters(d=5., T_cold=200., T_hot=800.)
    u = solver.set_initial_condition()
    assert(u[0, 0] == 200)

