import numpy as np
import sympy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib import cm

x, y, t = sp.symbols('x,y,t')

class Wave2D:

    def create_mesh(self, N, sparse=False):
        """Create 2D mesh and store in self.xij and self.yij"""
        # self.xji, self.yij = ...
        x = np.linspace(0, 1, N+1)
        y = np.linspace(0, 1, N+1)
        self.xij, self.yij = np.meshgrid(x, y, sparse=sparse)
        return self.xij, self.yij

    def D2(self, N):
        """Return second order differentiation matrix"""
        assert N >= 4
        D = sparse.diags([1, -2, 1], [-1, 0, 1],
                         (N + 1, N + 1), 'lil')
        D[0, :4] = 2, -5, 4, -1
        D[-1, -4:] = -1, 4, -5, 2
        return D

    @property
    def w(self):
        """Return the dispersion coefficient"""
        # Assuming k = m times π
        norm_k = np.pi * np.sqrt(self.mx**2 + self.my**2)
        return self.c * norm_k

    def ue(self, mx, my):
        """Return the exact standing wave"""
        return sp.sin(mx*sp.pi*x)*sp.sin(my*sp.pi*y)*sp.cos(self.w*t)

    def initialize(self, N, mx, my):
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """
        Unp1, Un, Unm1 = np.zeros((3, N+1, N+1))
        # Exact at t=0
        Unm1[:] = sp.lambdify((t, x, y), self.ue(self.mx, self.my))(0.0, self.xij, self.yij)
        # Exact at t=∆t
        Un[:] = sp.lambdify((t, x, y), self.ue(self.mx, self.my))(self.dt, self.xij, self.yij)
        return Unp1, Un, Unm1

    @property
    def dt(self):
        """Return the time step"""
        return self.cfl * self.dx / self.c

    def l2_error(self, u, t0):
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        uj = sp.lambdify((t, x, y), self.ue(self.mx, self.my))(t0, self.xij, self.yij)
        return np.sqrt(self.dx * self.dy * np.sum((uj - u)**2))

    def apply_bcs(self, Unp1):
        # Boundary
        Unp1[0, :] = 0.0
        Unp1[-1, :] = 0.0
        Unp1[:, -1] = 0.0
        Unp1[:, 0] = 0.0
        return Unp1

    def __call__(self, N, Nt, cfl=0.5, c=1.0, mx=3, my=3, store_data=-1):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """
        self.c = c
        self.N = N
        self.Nt = Nt
        self.cfl = cfl
        self.mx = mx
        self.my = my
        self.store_data = store_data
        self.dx = self.dy = 1.0 / N

        # Run the solver
        xij, yij = self.create_mesh(N)
        Dx = Dy = self.D2(N) / self.dx**2
        Unp1, Un, Unm1 = self.initialize(N, mx, my)
        # Solve for U at n=0 assuming Unm1 = Unp1
        # But this is introduces too much error for the Dirichlet BC
        # Un[:] = Unm1[:] + 0.5 * (c * self.dt)**2 * (Dx @ Unm1 + Unm1 @ Dy.T)
        plotdata = {0: Unm1.copy()}
        if store_data == 1:
            plotdata[1] = Un.copy()
        for n in range(1, Nt):
            # March
            Unp1[:] = 2 * Un - Unm1 + (c * self.dt)**2 * (Dx @ Un + Un @ Dy.T)
            # Boundary
            Unp1 = self.apply_bcs(Unp1)
            # Store
            if store_data > 0 and n % store_data == 0:
                plotdata[n] = Unp1.copy()
            # Swap
            Unm1[:] = Un
            Un[:] = Unp1
        # When n=Nt - 1, Unp1 was U[n], so after swapping Un has become U[Nt]
        plotdata[Nt] = Un.copy()
        if store_data == -1:
            return (self.dx, [self.l2_error(Un, Nt * self.dt)])
        else:
            return xij, yij, plotdata

    def convergence_rates(self, m=4, cfl=0.1, Nt=10, mx=3, my=3):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for m in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err[-1])
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)

class Wave2D_Neumann(Wave2D):

    def D2(self, N):
        """Differentiation matrix with inbuilt Neumann BCs"""
        assert N >= 4
        D = sparse.diags([1, -2, 1], [-1, 0, 1],
                         (N + 1, N + 1), 'lil')
        D[0, :4] = -2, +2, 0, 0
        D[-1, -4:] = 0, 0, +2, -2
        return D

    def ue(self, mx, my):
        """Return the exact standing wave"""
        return sp.cos(mx*sp.pi*x)*sp.cos(my*sp.pi*y)*sp.cos(self.w*t)

    def apply_bcs(self, Unp1):
        """Neumann BCs"""
        return Unp1

def test_convergence_wave2d():
    sol = Wave2D()
    r, E, h = sol.convergence_rates(m=5, mx=2, my=3)
    assert abs(r[-1]-2) < 1e-2

def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, E, h = solN.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 0.05

def test_exact_wave2d():
    sol = Wave2D()
    r, E, h = sol.convergence_rates(mx=3, my=3, cfl=1.0/np.sqrt(2))
    assert np.min(E) < 1e-12
    solN = Wave2D_Neumann()
    r, E, h = solN.convergence_rates(mx=3, my=3, cfl=1.0/np.sqrt(2))
    assert np.min(E) < 1e-12

if __name__ == "__main__":
    test_convergence_wave2d()
    test_exact_wave2d()
    test_convergence_wave2d_neumann()
