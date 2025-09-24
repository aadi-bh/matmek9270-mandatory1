import numpy as np
import sympy as sp
import scipy.sparse as sparse

x, y = sp.symbols('x,y')

class Poisson2D:
    r"""Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), in [0, L]^2

    where L is the length of the domain in both x and y directions.
    Dirichlet boundary conditions are used for the entire boundary.
    The Dirichlet values depend on the chosen manufactured solution.

    """

    def __init__(self, L, ue):
        """Initialize Poisson solver for the method of manufactured solutions

        Parameters
        ----------
        L : number
            The length of the domain in both x and y directions
        ue : Sympy function
            The analytical solution used with the method of manufactured solutions.
            ue is used to compute the right hand side function f.
        """
        self.L = L
        self.Lx = L
        self.Ly = L
        self.ue = ue
        self.f = sp.diff(self.ue, x, 2)+sp.diff(self.ue, y, 2)

    def create_mesh(self, N):
        """Create 2D mesh and store in self.xij and self.yij"""
        # Number of intervals
        self.Nx, self.Ny = N, N
        x = np.linspace(0, self.Lx, self.Nx+1)
        y = np.linspace(0, self.Ly, self.Ny+1)
        self.dx = abs(x[1] - x[0])
        self.dy = abs(y[1] - y[0])
        mesh = np.meshgrid(x, y, indexing='ij')
        self.xij, self.yij = mesh
        # self.xij, self.yij ...
        return self.xij, self.yij

    def D2(self, N):
        """Return second order differentiation matrix"""
        assert N >= 4
        D = sparse.diags([1, -2, 1], [-1, 0, 1], 
                         (N + 1, N + 1), 'lil')
        D[0, :4] = 2, -5, 4, -1
        D[-1, -4:] = -1, 4, -5, 2
        return D

    def laplace(self, dx, dy, Nx, Ny):
        """Return vectorized Laplace operator"""
        
        D2x = 1/self.dx**2 * self.D2(Nx)
        D2y = 1/self.dy**2 * self.D2(Ny)
        return (sparse.kron(D2x, sparse.eye(Ny+1)) + 
                sparse.kron(sparse.eye(Nx + 1), D2y))

    def get_boundary_indices(self, Nx, Ny):
        """Return indices of vectorized matrix (with `ravel()`) that belongs to the boundary"""
        B = np.ones((Nx+1, Ny+1))
        B[1:-1, 1:-1] = 0.0
        indices = np.where(B.ravel() == 1.0)[0]
        return indices

    def assemble(self):
        """Return assembled matrix A and right hand side vector b"""
        # LHS is the laplace operator
        A = self.laplace(self.dx, self.dy, self.Nx, self.Ny)
        # RHS is f
        b = sp.lambdify((x,y), self.f)(self.xij, self.yij)
        b = b.ravel()

        # Set boundary to ue
        bnds = self.get_boundary_indices(self.Nx, self.Ny)
        F = sp.lambdify((x, y), self.ue)(self.xij, self.yij).ravel()
        b[bnds] = F[bnds]
        # Set A to Dirichlet BC
        A = A.tolil()
        A[bnds, :] = 0.0
        for i in bnds:
            A[i,i] = 1.0
        A = A.tocsr()
        return A, b

    def l2_error(self, u):
        """Return l2-error norm"""
        uj = sp.lambdify((x, y), self.ue)(self.xij, self.yij)
        return np.sqrt(self.dx * self.dy * np.sum((uj - u)**2))

    def __call__(self, N):
        """Solve Poisson's equation.

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction

        Returns
        -------
        The solution as a Numpy array

        """
        self.create_mesh(N)
        A, b = self.assemble()
        self.U = sparse.linalg.spsolve(A, b.flatten()).reshape((N+1, N+1))
        return self.U

    def convergence_rates(self, m=6):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretization levels to use

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
            u = self(N0)
            E.append(self.l2_error(u))
            h.append(self.dx)
            N0 *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)

    def lagrange_basis(xj, x = x):
        """Construct a Lagrange basis for the given points

        Parameters
        ----------
        xj: array
            Interpolation points (nodes)
        x: Sympy symbol
        
        Returns
        -------
        List of Sympy polynomials
        """
        n = len(xj)
        ell = []
        numerator_all = sp.Mul(*[x- xji for xji in xj])
        for i in range(n):
            num = numerator_all / (x - xj[i])
            den = Mul(*[(xj[i] - xj[j]) for j in range(n) if i != j])
            ell.append(num/den)
        return ell
    def lagrange_function(u, basisx, basisy):
        """Return Lagrange polynomial

        Parameters
        ---------
        u: array
            Mesh function values
        basisx: Matrix of Lagrange basis functions in x-direction
        basisy: Matrix of Lagrange basis functions in y-direction
        """
        f = 0.0
        N, M = u.shape
        for i in range(N):
            for j in range(M):
                f += basisx[i] * basisy[j] * u[i, j]
        return f
    def eval(self, xf, yf):
        """Return u(xf, yf)

        Parameters
        ----------
        xf, yf : numbers
            The coordinates for evaluation

        Returns
        -------
        The value of u(xf, yf)

        """
        # Which values to interpolate from?
        radius = self.dx * 2
        xnbhd = np.asarray(np.abs(xf - self.xij) < radius).nonzero()
        ynbhd = np.asarray(np.abs(yf - self.yij) < radius).nonzero()
        nbhd = np.logical_and(xnbhd, ynbhd)
        basisx = self.lagrange_basis(self.xij)
        basisy = self.lagrange_basis(self.yij)
        L = self.lagrange_function(u, basisx, basisy)
        return L.subs({x: xf, y: fy})
        # self.U interpolated at x, y

def test_convergence_poisson2d():
    # This exact solution is NOT zero on the entire boundary
    ue = sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    sol = Poisson2D(1, ue)
    r, E, h = sol.convergence_rates()
    assert abs(r[-1]-2) < 1e-2

def test_interpolation():
    ue = sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    sol = Poisson2D(1, ue)
    U = sol(100)
    assert abs(sol.eval(0.52, 0.63) - ue.subs({x: 0.52, y: 0.63}).n()) < 1e-3
    assert abs(sol.eval(sol.h/2, 1-sol.h/2) - ue.subs({x: sol.h/2, y: 1-sol.h/2}).n()) < 1e-3

if __name__ == "__main__":
    test_interpolation()
    test_convergence_poisson2d()
