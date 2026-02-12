import torch
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt

# ==============================================================================
# 0. CONFIGURATION & DEVICE
# ==============================================================================
# Use non-interactive backend to save files without a display
matplotlib.use('Agg')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on: {device}")

# Fix for deprecated set_default_tensor_type
if device.type == 'cuda':
    torch.set_default_dtype(torch.float32)
    torch.set_default_device(device)
else:
    torch.set_default_dtype(torch.float32)

torch.manual_seed(42)
np.random.seed(42)

# ==============================================================================
# 1. SOLVERS (Updated with Gradient Prediction)
# ==============================================================================

class FastLSQSolver:
    """Fast-LSQ: Random Fourier Features with sin activation"""
    def __init__(self, input_dim, output_dim=1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W_list = []
        self.b_list = []
        self.beta = None
        self.name = "Fast-LSQ (sin)"

    def add_block(self, hidden_size=500, scale=1.0):
        # Generate standard normal weights [D, H]
        W = torch.randn(self.input_dim, hidden_size, device=device)

        # --- MODIFICATION START: Handle Anisotropic Scaling ---
        if isinstance(scale, list) or isinstance(scale, np.ndarray):
            # If scale is a list [sx, sy, st], reshape to [D, 1] for broadcasting
            s_tensor = torch.tensor(scale, device=device, dtype=torch.float32).unsqueeze(1)
            W = W * s_tensor
        else:
            # Standard scalar scaling
            W = W * scale
        # --- MODIFICATION END ---

        b = (torch.rand(1, hidden_size, device=device) * 2 * np.pi)
        self.W_list.append(W)
        self.b_list.append(b)

    def get_features(self, x):
        """Returns H, dH, ddH"""
        Hs, dHs, ddHs = [], [], []
        for W, b in zip(self.W_list, self.b_list):
            Z = x @ W + b
            H = torch.sin(Z)
            H_cos = torch.cos(Z)

            # dH shape: [N, D, Hidden]
            # derivative of sin(w*x) is w*cos(w*x)
            dH = H_cos.unsqueeze(1) * W.unsqueeze(0)

            # ddH shape: [N, D, Hidden] (diagonal of Hessian)
            # derivative of w*cos(w*x) is -w^2*sin(w*x)
            ddH = -H.unsqueeze(1) * (W**2).unsqueeze(0)

            Hs.append(H)
            dHs.append(dH)
            ddHs.append(ddH)

        return torch.cat(Hs, -1), torch.cat(dHs, -1), torch.cat(ddHs, -1)

    def predict_with_grad(self, x):
        """Returns value u and gradient vector field du/dx"""
        H, dH, _ = self.get_features(x)
        u = H @ self.beta
        grad_u = torch.einsum('idh,h->id', dH, self.beta.squeeze())
        return u, grad_u


class PIELMSolver:
    """PIELM: Physics-Informed ELM with tanh activation"""
    def __init__(self, input_dim, output_dim=1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W_list = []
        self.b_list = []
        self.beta = None
        self.name = "PIELM (tanh)"

    def add_block(self, hidden_size=500, scale=1.0):
        # Generate random weights Uniform[-1, 1]
        W_base = torch.rand(self.input_dim, hidden_size, device=device) * 2 - 1

        # --- MODIFICATION: Handle Anisotropic Scaling (Fixes TypeError) ---
        if isinstance(scale, list) or isinstance(scale, np.ndarray):
            # Broadcast scale list [sx, sy, st] -> [D, 1]
            s_tensor = torch.tensor(scale, device=device, dtype=torch.float32).unsqueeze(1)
            W = W_base * s_tensor

            # Scale bias by the max scale to keep activation active
            # or use a mean. Usually max or mean of scales is safe for bias.
            b_scale = max(scale)
        else:
            W = W_base * scale
            b_scale = scale
        # ------------------------------------------------------------------

        b = (torch.rand(1, hidden_size, device=device) * 2 - 1) * b_scale
        self.W_list.append(W)
        self.b_list.append(b)

    def get_features(self, x):
        Hs, dHs, ddHs = [], [], []
        for W, b in zip(self.W_list, self.b_list):
            Z = x @ W + b
            H = torch.tanh(Z)
            sech2 = 1 - H**2

            d_tanh = sech2
            dd_tanh = -2 * H * sech2

            dH = d_tanh.unsqueeze(1) * W.unsqueeze(0)
            ddH = dd_tanh.unsqueeze(1) * (W**2).unsqueeze(0)

            Hs.append(H)
            dHs.append(dH)
            ddHs.append(ddH)

        return torch.cat(Hs, -1), torch.cat(dHs, -1), torch.cat(ddHs, -1)

    def predict_with_grad(self, x):
        H, dH, _ = self.get_features(x)
        u = H @ self.beta
        grad_u = torch.einsum('idh,h->id', dH, self.beta.squeeze())
        return u, grad_u

# ==============================================================================
# 2. PROBLEM DEFINITIONS (With Exact Gradients)
# ==============================================================================

class PoissonND:
    def __init__(self):
        self.name = "Poisson 5D"
        self.dim = 5

    def exact(self, x):
        return torch.sum(torch.sin(np.pi/2 * x), dim=1, keepdim=True)

    def exact_grad(self, x):
        # du/dx_i = (pi/2) * cos(pi/2 * x_i)
        # Returns [N, 5]
        return (np.pi/2) * torch.cos(np.pi/2 * x)

    def source(self, x):
        return (np.pi**2 / 4) * torch.sum(torch.sin(np.pi/2 * x), dim=1, keepdim=True)

    def get_train_data(self, n_pde=10000, n_bc=2000):
        x_pde = torch.rand(n_pde, self.dim, device=device)
        f_pde = self.source(x_pde)
        x_bc = torch.rand(n_bc, self.dim, device=device)
        mask_dim = torch.randint(0, self.dim, (n_bc,), device=device)
        mask_val = torch.randint(0, 2, (n_bc,), device=device).float()
        for i in range(n_bc):
            x_bc[i, mask_dim[i]] = mask_val[i]
        u_bc = self.exact(x_bc)
        return x_pde, [(x_bc, u_bc)], f_pde

    def build(self, slv, x_pde, bcs, f_pde):
        _, _, ddH = slv.get_features(x_pde)
        A = -torch.sum(ddH, dim=1)
        b = f_pde
        As, bs = [A], [b]
        for (pts, vals) in bcs:
            h, _, _ = slv.get_features(pts)
            As.append(h * 100.0)
            bs.append(vals * 100.0)
        return torch.cat(As), torch.cat(bs)

    def get_test_points(self, n=10000):
        return torch.rand(n, self.dim, device=device)


class HeatND:
    def __init__(self):
        self.name = "Heat 5D"
        self.dim = 6
        self.d = 5
        self.k = 1.0 / 5.0

    def exact(self, x):
        r2 = torch.sum(x[:, 0:5]**2, dim=1, keepdim=True)
        t = x[:, 5:6]
        return torch.exp(0.5 * r2 + t)

    def exact_grad(self, x):
        # u = exp(0.5 r^2 + t)
        # du/dx_i = x_i * u (for i=1..5)
        # du/dt = u
        u = self.exact(x)
        grad_spatial = x[:, 0:5] * u
        grad_time = u
        return torch.cat([grad_spatial, grad_time], dim=1)

    def source(self, x):
        r2 = torch.sum(x[:, 0:5]**2, dim=1, keepdim=True)
        return -(1.0/self.d) * r2 * self.exact(x)

    def sample_sphere_time(self, n):
        pts = torch.randn(n, 5, device=device)
        pts = pts / torch.norm(pts, dim=1, keepdim=True)
        r = torch.rand(n, 1, device=device) ** (1/5.0)
        spatial = pts * r
        time = torch.rand(n, 1, device=device)
        return torch.cat([spatial, time], dim=1)

    def get_train_data(self, n_pde=10000, n_bc=2000):
        x_pde = self.sample_sphere_time(n_pde)
        f_pde = self.source(x_pde)
        x_ic_space = self.sample_sphere_time(n_bc)[:, 0:5]
        x_ic = torch.cat([x_ic_space, torch.zeros(n_bc, 1, device=device)], 1)
        u_ic = self.exact(x_ic)
        x_bc_space = torch.randn(n_bc, 5, device=device)
        x_bc_space = x_bc_space / torch.norm(x_bc_space, dim=1, keepdim=True)
        t_bc = torch.rand(n_bc, 1, device=device)
        x_bc = torch.cat([x_bc_space, t_bc], 1)
        g_bc = self.exact(x_bc)
        return x_pde, [(x_ic, u_ic, 'dirichlet'), (x_bc, g_bc, 'neumann_n')], f_pde

    def build(self, slv, x_pde, bcs, f_pde):
        _, dH, ddH = slv.get_features(x_pde)
        lap = torch.sum(ddH[:, 0:5, :], dim=1)
        u_t = dH[:, 5, :]
        A = u_t - self.k * lap
        b = f_pde
        As, bs = [A], [b]
        for (pts, vals, type_) in bcs:
            h, dh, _ = slv.get_features(pts)
            w = 100.0
            if type_ == 'dirichlet':
                As.append(h * w)
            elif type_ == 'neumann_n':
                neumann_term = torch.zeros_like(h)
                for i in range(5):
                    neumann_term += pts[:, i:i+1] * dh[:, i, :]
                As.append(neumann_term * w)
            bs.append(vals * w)
        return torch.cat(As), torch.cat(bs)

    def get_test_points(self, n=10000):
        return self.sample_sphere_time(n)


class Wave1D:
    def __init__(self):
        self.name = "Wave 1D"
        self.dim = 2
        self.c2 = 4.0

    def exact(self, x):
        xv, tv = x[:,0:1], x[:,1:2]
        return torch.sin(np.pi*xv)*torch.cos(2*np.pi*tv) + 0.5*torch.sin(4*np.pi*xv)*torch.cos(8*np.pi*tv)

    def exact_grad(self, x):
        xv, tv = x[:,0:1], x[:,1:2]
        # u_x
        ux = np.pi * torch.cos(np.pi*xv)*torch.cos(2*np.pi*tv) + \
             0.5 * 4*np.pi * torch.cos(4*np.pi*xv)*torch.cos(8*np.pi*tv)
        # u_t
        ut = -2*np.pi * torch.sin(np.pi*xv)*torch.sin(2*np.pi*tv) - \
             0.5 * 8*np.pi * torch.sin(4*np.pi*xv)*torch.sin(8*np.pi*tv)
        return torch.cat([ux, ut], dim=1)

    def get_train_data(self, n_pde=5000, n_bc=1000):
        x_pde = torch.rand(n_pde, 2, device=device)
        x_ic = torch.cat([torch.rand(n_bc,1, device=device), torch.zeros(n_bc,1, device=device)], 1)
        u_ic = torch.sin(np.pi*x_ic[:,0:1]) + 0.5*torch.sin(4*np.pi*x_ic[:,0:1])
        ut_ic = torch.zeros_like(u_ic)
        t_bc = torch.rand(n_bc, 1, device=device)
        x_bc_l = torch.cat([torch.zeros_like(t_bc), t_bc], 1)
        x_bc_r = torch.cat([torch.ones_like(t_bc), t_bc], 1)
        u_bc = torch.zeros_like(t_bc)
        return x_pde, [(x_ic, u_ic, 'dirichlet'), (x_ic, ut_ic, 'neumann_t'),
                       (x_bc_l, u_bc, 'dirichlet'), (x_bc_r, u_bc, 'dirichlet')]

    def build(self, slv, x_pde, bcs):
        _, _, ddH = slv.get_features(x_pde)
        A = ddH[:,1,:] - self.c2*ddH[:,0,:]
        b = torch.zeros(len(x_pde), 1, device=device)
        As, bs = [A], [b]
        for (pts, vals, type_) in bcs:
            h, dh, _ = slv.get_features(pts)
            w = 100.0
            if type_ == 'dirichlet':
                As.append(h * w)
            elif type_ == 'neumann_t':
                As.append(dh[:,1,:] * w)
            bs.append(vals * w)
        return torch.cat(As), torch.cat(bs)

    def get_test_points(self, n=10000):
        return torch.rand(n, self.dim, device=device)


class Wave2D_MS:
    """
    Wave 2D Multi-Scale with Time Normalization and Frequency Compensation.
    """
    def __init__(self):
        self.name = "Wave 2D-MS"
        self.dim = 3
        self.a2 = 2.0

        # 1. Normalization Constant (Time domain 0 to 100)
        self.t_max = 100.0

        # 2. Anisotropic Scale Multipliers [Scale_X, Scale_Y, Scale_T]
        # Since t is normalized by 100, the wave frequency in 'normalized space'
        # is 100x higher. We need high-variance weights for the 3rd dim (Time).
        self.scale_multipliers = [1.0, 1.0, 300.0]

    def exact(self, x_in):
        # x_in is normalized [0,1]. Denormalize time for physics formula.
        xv = x_in[:, 0:1]
        yv = x_in[:, 1:2]
        tv = x_in[:, 2:3] * self.t_max

        # Frequency omega = pi * sqrt(1 + a2) = pi * sqrt(3) approx 5.44
        omega = np.pi * np.sqrt(1 + self.a2)
        return torch.sin(np.pi*xv) * torch.sin(np.pi*yv) * torch.cos(omega*tv)

    def exact_grad(self, x_in):
        xv = x_in[:, 0:1]
        yv = x_in[:, 1:2]
        tv = x_in[:, 2:3] * self.t_max
        omega = np.pi * np.sqrt(1 + self.a2)

        # Derivatives wrt Physical Coordinates
        u_x = np.pi * torch.cos(np.pi*xv) * torch.sin(np.pi*yv) * torch.cos(omega*tv)
        u_y = np.pi * torch.sin(np.pi*xv) * torch.cos(np.pi*yv) * torch.cos(omega*tv)
        u_t_phys = -omega * torch.sin(np.pi*xv) * torch.sin(np.pi*yv) * torch.sin(omega*tv)

        # Chain rule: du/dt_norm = du/dt_phys * (dt_phys / dt_norm)
        # dt_phys / dt_norm = t_max
        u_t_norm = u_t_phys * self.t_max

        return torch.cat([u_x, u_y, u_t_norm], dim=1)

    def get_train_data(self, n_pde=5000, n_bc=1000):
        # Sample in Normalized Domain [0, 1]
        x_pde = torch.rand(n_pde, 3, device=device)

        # IC: t_norm = 0
        x_ic = torch.cat([torch.rand(n_bc, 2, device=device), torch.zeros(n_bc, 1, device=device)], 1)
        u_ic = self.exact(x_ic)
        ut_ic = torch.zeros(n_bc, 1, device=device)

        # BC: x=0,1 y=0,1 (t_norm in 0..1)
        x_bc = torch.rand(n_bc, 3, device=device)
        mask = torch.randint(0, 4, (n_bc,), device=device)
        x_bc[mask==0, 0] = 0; x_bc[mask==1, 0] = 1
        x_bc[mask==2, 1] = 0; x_bc[mask==3, 1] = 1
        u_bc = self.exact(x_bc)

        return x_pde, [(x_ic, u_ic, 'dirichlet'), (x_ic, ut_ic, 'neumann_t'), (x_bc, u_bc, 'dirichlet')], None

    def build(self, slv, x_pde, bcs, f_pde_ignored):
        _, dH, ddH = slv.get_features(x_pde)

        u_xx = ddH[:, 0, :]
        u_yy = ddH[:, 1, :]
        u_tt_norm = ddH[:, 2, :]

        # PDE Operator in Normalized Coords:
        # u_tt_norm - (t_max^2)*(u_xx + a2*u_yy) = 0
        A = u_tt_norm - (self.t_max**2) * (u_xx + self.a2 * u_yy)
        b = torch.zeros(len(x_pde), 1, device=device)

        As, bs = [A], [b]
        w_bc = 1000.0 # High weight for boundaries

        for (pts, vals, type_) in bcs:
            h, dh, _ = slv.get_features(pts)
            if type_ == 'dirichlet':
                As.append(h * w_bc)
            elif type_ == 'neumann_t':
                As.append(dh[:, 2, :] * w_bc)
            bs.append(vals * w_bc)

        return torch.cat(As), torch.cat(bs)

    def get_test_points(self, n=2000):
        return torch.rand(n, 3, device=device)


class Burgers1D_Regression:
    """
    Scenario: Traveling Shock fitting
    Equation: u_t + u*u_x = nu * u_xx
    Exact Solution used: u(x,t) = 0.5 * (1 - tanh((x - 0.5t)/(4nu)))
    """
    def __init__(self):
        self.name = "Burgers (Shock)"
        self.dim = 2
        self.nu = 0.02 # Controls shock sharpness

    def exact(self, x_in):
        # Traveling wave solution
        xv, tv = x_in[:, 0:1], x_in[:, 1:2]
        z = (xv - 0.5 * tv) / (4 * self.nu)
        return 0.5 * (1 - torch.tanh(z))

    def exact_grad(self, x_in):
        # We need analytic derivatives for the error metric
        # u = 0.5 * (1 - tanh(z))
        # du/dz = -0.5 * sech^2(z)

        xv, tv = x_in[:, 0:1], x_in[:, 1:2]
        z = (xv - 0.5 * tv) / (4 * self.nu)

        tanh_z = torch.tanh(z)
        sech2_z = 1 - tanh_z**2
        du_dz = -0.5 * sech2_z

        # Chain rule:
        # du/dx = du/dz * dz/dx = du/dz * (1 / 4nu)
        # du/dt = du/dz * dz/dt = du/dz * (-0.5 / 4nu)

        dz_dx = 1.0 / (4 * self.nu)
        dz_dt = -0.5 / (4 * self.nu)

        du_dx = du_dz * dz_dx
        du_dt = du_dz * dz_dt

        return torch.cat([du_dx, du_dt], dim=1)

    def get_train_data(self, n_samples=5000):
        # Domain: x in [0, 1], t in [0, 1]
        x_pde = torch.rand(n_samples, 2, device=device)
        u_true = self.exact(x_pde)
        # Fitting data directly
        return x_pde, [(x_pde, u_true, 'data_fit')]

    def build(self, slv, x_pde, bcs):
        # A = H (Features), b = u (Target)
        As, bs = [], []
        for (pts, vals, _) in bcs:
            h, _, _ = slv.get_features(pts)
            As.append(h)
            bs.append(vals)
        return torch.cat(As), torch.cat(bs)

    def get_test_points(self, n=10000):
        return torch.rand(n, self.dim, device=device)
class KdV_Regression:
    """
    Korteweg-de Vries (KdV) Equation
    Equation: u_t + u*u_x + u_xxx = 0
    Scenario: Single Soliton traveling to the right.
    Exact Sol: u(x,t) = 3*c * sech^2( (sqrt(c)/2) * (x - c*t - x0) )
    """
    def __init__(self):
        self.name = "KdV (Soliton)"
        self.dim = 2
        self.c = 30.0   # Wave speed (higher = sharper peak)
        self.x0 = -1.0  # Starting offset

    def exact(self, x_in):
        xv, tv = x_in[:, 0:1], x_in[:, 1:2]

        # Argument for sech
        # z = (sqrt(c)/2) * (x - c*t - x0)
        sqrt_c = np.sqrt(self.c)
        z = (sqrt_c / 2.0) * (xv - self.c * tv - self.x0)

        # u = 3c * sech^2(z)
        # Note: torch has no sech, use 1/cosh
        sech_z = 1.0 / torch.cosh(z)
        u = 3 * self.c * (sech_z ** 2)

        return u

    def exact_grad(self, x_in):
        # Derivation for u = A * sech^2(z)
        # du/dz = 2*A*sech(z)*(-sech(z)tanh(z)) = -2*A * sech^2(z) * tanh(z)
        # du/dz = -2 * u * tanh(z)

        xv, tv = x_in[:, 0:1], x_in[:, 1:2]
        sqrt_c = np.sqrt(self.c)
        z = (sqrt_c / 2.0) * (xv - self.c * tv - self.x0)

        sech_z = 1.0 / torch.cosh(z)
        u = 3 * self.c * (sech_z ** 2)
        tanh_z = torch.tanh(z)

        # du/dz
        du_dz = -2.0 * u * tanh_z

        # Chain rule derivatives
        # z = k * (x - ct - x0)
        # dz/dx = k
        # dz/dt = -k*c
        k = sqrt_c / 2.0

        du_dx = du_dz * k
        du_dt = du_dz * (-k * self.c)

        return torch.cat([du_dx, du_dt], dim=1)

    def get_train_data(self, n_samples=5000):
        # Domain: x in [-2, 2], t in [0, 0.1] (short time for soliton)
        x_space = torch.rand(n_samples, 1, device=device) * 4 - 2
        t_time = torch.rand(n_samples, 1, device=device) * 0.1

        x_pde = torch.cat([x_space, t_time], dim=1)
        u_true = self.exact(x_pde)

        return x_pde, [(x_pde, u_true, 'data_fit')]

    def build(self, slv, x_pde, bcs):
        # Regression Mode: A = H, b = u
        As, bs = [], []
        for (pts, vals, _) in bcs:
            h, _, _ = slv.get_features(pts)
            As.append(h)
            bs.append(vals)
        return torch.cat(As), torch.cat(bs)

    def get_test_points(self, n=10000):
        x_space = torch.rand(n, 1, device=device) * 4 - 2
        t_time = torch.rand(n, 1, device=device) * 0.1
        return torch.cat([x_space, t_time], dim=1)

class ReactionDiffusion_Regression:
    """
    Reaction-Diffusion (Fisher's Equation)
    Equation: u_t - u_xx - u(1-u) = 0  (Assuming D=1, rho=1 for simplicity)
    Scenario: Traveling wavefront.
    """
    def __init__(self):
        self.name = "Reaction-Diffusion"
        self.dim = 2
        # Parameters for the standard analytic traveling wave
        self.rho = 1.0
        # The standard analytic solution usually assumes specific D/rho relation.
        # Common form: u = (1 + C * exp( z ))^-2
        # where z = sqrt(rho/6) * (x - 5/6 * sqrt(rho*6) * t)

    def exact(self, x_in):
        xv, tv = x_in[:, 0:1], x_in[:, 1:2]

        # Calculate z argument
        # alpha = sqrt(rho/6)
        alpha = np.sqrt(self.rho / 6.0)
        # c = (5/6) * sqrt(rho*6)
        c = (5.0 / 6.0) * np.sqrt(self.rho * 6.0)

        z = alpha * (xv - c * tv)

        # u = (1 + exp(z))^-2
        # For numerical stability with exp, we can use sigmoid forms or clamp,
        # but exact solution is robust in typical domains.
        term = 1.0 + torch.exp(z)
        u = term.pow(-2)

        return u

    def exact_grad(self, x_in):
        # Derivation:
        # u = (1 + e^z)^-2
        # Let E = e^z. u = (1+E)^-2
        # du/dz = -2(1+E)^-3 * E

        xv, tv = x_in[:, 0:1], x_in[:, 1:2]
        alpha = np.sqrt(self.rho / 6.0)
        c = (5.0 / 6.0) * np.sqrt(self.rho * 6.0)
        z = alpha * (xv - c * tv)

        E = torch.exp(z)
        base = 1.0 + E

        # du/dz
        du_dz = -2.0 * (base.pow(-3)) * E

        # Chain rule
        # dz/dx = alpha
        # dz/dt = -alpha * c

        du_dx = du_dz * alpha
        du_dt = du_dz * (-alpha * c)

        return torch.cat([du_dx, du_dt], dim=1)

    def get_train_data(self, n_samples=5000):
        # Domain: x in [-10, 10] to see the full wave, t in [0, 1]
        x_space = torch.rand(n_samples, 1, device=device) * 20 - 10
        t_time = torch.rand(n_samples, 1, device=device)

        x_pde = torch.cat([x_space, t_time], dim=1)
        u_true = self.exact(x_pde)

        return x_pde, [(x_pde, u_true, 'data_fit')]

    def build(self, slv, x_pde, bcs):
        # Regression Mode
        As, bs = [], []
        for (pts, vals, _) in bcs:
            h, _, _ = slv.get_features(pts)
            As.append(h)
            bs.append(vals)
        return torch.cat(As), torch.cat(bs)

    def get_test_points(self, n=10000):
        x_space = torch.rand(n, 1, device=device) * 20 - 10
        t_time = torch.rand(n, 1, device=device)
        return torch.cat([x_space, t_time], dim=1)

class Helmholtz2D:
    """
    Helmholtz Equation (2D)
    Equation: u_xx + u_yy + k^2 u = f
    Exact Solution: u = sin(k*x) * sin(k*y)
    Note: This is LINEAR, so we can use the Solver mode!
    """
    def __init__(self):
        self.name = "Helmholtz 2D"
        self.dim = 2
        self.k = 10.0 # High frequency parameter

    def exact(self, x):
        return torch.sin(self.k * x[:,0:1]) * torch.sin(self.k * x[:,1:2])

    def exact_grad(self, x):
        k = self.k
        sx = torch.sin(k * x[:,0:1])
        cx = torch.cos(k * x[:,0:1])
        sy = torch.sin(k * x[:,1:2])
        cy = torch.cos(k * x[:,1:2])

        du_dx = k * cx * sy
        du_dy = k * sx * cy
        return torch.cat([du_dx, du_dy], dim=1)

    def source(self, x):
        # f = u_xx + u_yy + k^2 u
        # u_xx = -k^2 * u
        # u_yy = -k^2 * u
        # f = -k^2 u - k^2 u + k^2 u = -k^2 u
        return -(self.k**2) * self.exact(x)

    def get_train_data(self, n_pde=10000, n_bc=2000):
        x_pde = torch.rand(n_pde, 2, device=device)
        f_pde = self.source(x_pde)

        # Boundary: box [0,1]x[0,1]
        # Generate random points on boundary
        n_side = n_bc // 4
        x_bc_list = []
        # Left (x=0), Right (x=1), Bottom (y=0), Top (y=1)
        y_rand = torch.rand(n_side, 1, device=device)
        x_rand = torch.rand(n_side, 1, device=device)
        zeros = torch.zeros(n_side, 1, device=device)
        ones = torch.ones(n_side, 1, device=device)

        x_bc_list.append(torch.cat([zeros, y_rand], 1)) # x=0
        x_bc_list.append(torch.cat([ones, y_rand], 1))  # x=1
        x_bc_list.append(torch.cat([x_rand, zeros], 1)) # y=0
        x_bc_list.append(torch.cat([x_rand, ones], 1))  # y=1

        x_bc = torch.cat(x_bc_list, 0)
        u_bc = self.exact(x_bc)

        return x_pde, [(x_bc, u_bc, 'dirichlet')], f_pde

    def build(self, slv, x_pde, bcs, f_pde):
        # Solver Mode
        _, _, ddH = slv.get_features(x_pde)
        # Laplacian: u_xx + u_yy = sum(ddH)
        lap = torch.sum(ddH, dim=1)

        # Operator A: lap + k^2 * Identity
        # We need H for the Identity part
        H, _, _ = slv.get_features(x_pde)

        A = lap + (self.k**2) * H
        b = f_pde

        As, bs = [A], [b]
        for (pts, vals, type_) in bcs:
            h, _, _ = slv.get_features(pts)
            w = 100.0
            As.append(h * w)
            bs.append(vals * w)

        return torch.cat(As), torch.cat(bs)

    def get_test_points(self, n=10000):
        return torch.rand(n, self.dim, device=device)

class Maxwell2D_TM:
    """
    Maxwell Equations (2D TM Mode in a PEC Cavity)
    Solving for Electric Field Ez(x, y, t).
    Equation: u_tt = c^2 * (u_xx + u_yy)
    Domain: Unit Box [0,1]^2
    """
    def __init__(self):
        self.name = "Maxwell 2D (TM)"
        self.dim = 3 # x, y, t
        self.c = 1.0
        # Resonant mode (1,1)
        self.kx = np.pi
        self.ky = np.pi
        self.omega = self.c * np.sqrt(self.kx**2 + self.ky**2)

    def exact(self, x_in):
        xv, yv, tv = x_in[:,0:1], x_in[:,1:2], x_in[:,2:3]
        return torch.sin(self.kx * xv) * torch.sin(self.ky * yv) * torch.cos(self.omega * tv)

    def exact_grad(self, x_in):
        xv, yv, tv = x_in[:,0:1], x_in[:,1:2], x_in[:,2:3]

        # u = sin(kx x) sin(ky y) cos(w t)
        # du/dx = kx cos(kx x) sin(ky y) cos(w t)
        du_dx = self.kx * torch.cos(self.kx*xv) * torch.sin(self.ky*yv) * torch.cos(self.omega*tv)
        # du/dy = ky sin(kx x) cos(ky y) cos(w t)
        du_dy = self.ky * torch.sin(self.kx*xv) * torch.cos(self.ky*yv) * torch.cos(self.omega*tv)
        # du/dt = -w sin(kx x) sin(ky y) sin(w t)
        du_dt = -self.omega * torch.sin(self.kx*xv) * torch.sin(self.ky*yv) * torch.sin(self.omega*tv)

        return torch.cat([du_dx, du_dy, du_dt], dim=1)

    def get_train_data(self, n_pde=5000, n_bc=1000):
        # Domain: x,y in [0,1], t in [0,1]
        x_pde = torch.rand(n_pde, 3, device=device)

        # Initial Conditions (t=0)
        x_ic_space = torch.rand(n_bc, 2, device=device)
        x_ic_t0 = torch.zeros(n_bc, 1, device=device)
        x_ic = torch.cat([x_ic_space, x_ic_t0], 1)
        u_ic = self.exact(x_ic)
        # For wave equation we also need u_t(t=0) = 0 (since cos(wt)' -> -sin(0)=0)
        ut_ic = torch.zeros_like(u_ic)

        # Boundary Conditions (PEC Walls -> E_z = 0)
        # 4 walls: x=0, x=1, y=0, y=1
        x_bc_list = []
        n_wall = n_bc // 4
        # Random time and other coordinate
        r_t = torch.rand(n_wall, 1, device=device)
        r_s = torch.rand(n_wall, 1, device=device)
        zeros = torch.zeros(n_wall, 1, device=device)
        ones = torch.ones(n_wall, 1, device=device)

        # x=0, x=1
        x_bc_list.append(torch.cat([zeros, r_s, r_t], 1))
        x_bc_list.append(torch.cat([ones, r_s, r_t], 1))
        # y=0, y=1
        x_bc_list.append(torch.cat([r_s, zeros, r_t], 1))
        x_bc_list.append(torch.cat([r_s, ones, r_t], 1))

        x_bc = torch.cat(x_bc_list, 0)
        u_bc = torch.zeros(len(x_bc), 1, device=device) # PEC

        return x_pde, [(x_ic, u_ic, 'dirichlet'),
                       (x_ic, ut_ic, 'neumann_t'),
                       (x_bc, u_bc, 'dirichlet')], None

    def build(self, slv, x_pde, bcs, f_pde_ignored):
        _, dH, ddH = slv.get_features(x_pde)

        # Wave Equation Operator: u_tt - c^2(u_xx + u_yy) = 0
        u_xx = ddH[:, 0, :]
        u_yy = ddH[:, 1, :]
        u_tt = ddH[:, 2, :]

        A = u_tt - (self.c**2) * (u_xx + u_yy)
        b = torch.zeros(len(x_pde), 1, device=device)

        As, bs = [A], [b]
        for (pts, vals, type_) in bcs:
            h, dh, _ = slv.get_features(pts)
            w = 100.0 # Weighting
            if type_ == 'dirichlet':
                As.append(h * w)
            elif type_ == 'neumann_t':
                As.append(dh[:, 2, :] * w) # Time derivative
            bs.append(vals * w)

        return torch.cat(As), torch.cat(bs)

    def get_test_points(self, n=2000):
        return torch.rand(n, 3, device=device)

class SineGordon_Regression:
    """Fixed Version: Explicitly includes get_train_data"""
    def __init__(self):
        self.name = "Sine-Gordon"
        self.dim = 2
        self.omega = 0.5

    def exact(self, x_in):
        xv, tv = x_in[:, 0:1], x_in[:, 1:2]
        w = self.omega
        denom_factor = np.sqrt(1 - w**2)
        num = denom_factor * torch.sin(w * tv)
        denom = w * torch.cosh(denom_factor * xv)
        return 4.0 * torch.atan(num / denom)

    def exact_grad(self, x_in):
        xv, tv = x_in[:, 0:1], x_in[:, 1:2]
        w = self.omega
        k = np.sqrt(1 - w**2)

        sin_wt = torch.sin(w * tv)
        cos_wt = torch.cos(w * tv)
        cosh_kx = torch.cosh(k * xv)
        sinh_kx = torch.sinh(k * xv)

        num = k * sin_wt
        denom = w * cosh_kx
        A = num / denom # Argument of atan

        # d(atan(A))/dx = 1/(1+A^2) * dA/dx
        du_dA = 4.0 / (1.0 + A**2)

        dA_dx = num * (-1.0 / (denom**2)) * (w * k * sinh_kx)
        dA_dt = (1.0 / denom) * (k * w * cos_wt)

        return torch.cat([du_dA * dA_dx, du_dA * dA_dt], dim=1)

    def get_train_data(self, n_samples=5000):
        # Domain: x in [-10, 10], t in [0, 20]
        x_space = torch.rand(n_samples, 1, device=device) * 20 - 10
        t_time = torch.rand(n_samples, 1, device=device) * 20
        x_pde = torch.cat([x_space, t_time], dim=1)
        u_true = self.exact(x_pde)
        # Regression Mode: Just return points and targets
        return x_pde, [(x_pde, u_true, 'data_fit')]

    def build(self, slv, x_pde, bcs):
        As, bs = [], []
        for (pts, vals, _) in bcs:
            h, _, _ = slv.get_features(pts)
            As.append(h)
            bs.append(vals)
        return torch.cat(As), torch.cat(bs)

    def get_test_points(self, n=2000):
        x_space = torch.rand(n, 1, device=device) * 20 - 10
        t_time = torch.rand(n, 1, device=device) * 20
        return torch.cat([x_space, t_time], dim=1)
class KleinGordon_Regression:
    """
    Nonlinear Klein-Gordon Equation
    Equation: u_tt - u_xx + u^2 = f
    Test Case: Manufactured smooth oscillatory solution.
    Why it's fair: Unlike Burgers (shocks), this tests nonlinear fitting
    on SMOOTH functions, where Fast-LSQ should theoretically win.
    """
    def __init__(self):
        self.name = "Klein-Gordon (NL)"
        self.dim = 2

    def exact(self, x_in):
        # A smooth, wave-like solution
        xv, tv = x_in[:, 0:1], x_in[:, 1:2]
        return torch.sin(np.pi * xv) * torch.cos(2 * np.pi * tv)

    def exact_grad(self, x_in):
        xv, tv = x_in[:, 0:1], x_in[:, 1:2]

        # u = sin(pi x) cos(2 pi t)
        # du/dx = pi cos(pi x) cos(2 pi t)
        du_dx = np.pi * torch.cos(np.pi * xv) * torch.cos(2 * np.pi * tv)

        # du/dt = -2pi sin(pi x) sin(2 pi t)
        du_dt = -2 * np.pi * torch.sin(np.pi * xv) * torch.sin(2 * np.pi * tv)

        return torch.cat([du_dx, du_dt], dim=1)

    def get_train_data(self, n_samples=5000):
        # Domain: x in [-1, 1], t in [0, 1]
        x_space = torch.rand(n_samples, 1, device=device) * 2 - 1
        t_time = torch.rand(n_samples, 1, device=device)
        x_pde = torch.cat([x_space, t_time], dim=1)

        u_true = self.exact(x_pde)

        # Return in Regression format
        return x_pde, [(x_pde, u_true, 'data_fit')]

    def build(self, slv, x_pde, bcs):
        # Standard Regression Build
        As, bs = [], []
        for (pts, vals, _) in bcs:
            h, _, _ = slv.get_features(pts)
            As.append(h)
            bs.append(vals)
        return torch.cat(As), torch.cat(bs)

    def get_test_points(self, n=2000):
        x_space = torch.rand(n, 1, device=device) * 2 - 1
        t_time = torch.rand(n, 1, device=device)
        return torch.cat([x_space, t_time], dim=1)

class NavierStokes2D_Kovasznay:
    """
    Navier-Stokes Equation (Steady State) - Kovasznay Flow
    This is an exact analytic solution to NS, often used as a benchmark
    because Lid-Driven flow has no closed-form solution.

    Domain: [-0.5, 1.0] x [-0.5, 1.5]
    """
    def __init__(self):
        self.name = "NS (Kovasznay)"
        self.dim = 2 # x, y (Steady state)
        self.Re = 20.0
        # Kovasznay parameters
        self.lam = 0.5 * self.Re - np.sqrt(0.25 * self.Re**2 + 4 * np.pi**2)

    def exact(self, x_in):
        # Returns u-velocity component
        xv, yv = x_in[:, 0:1], x_in[:, 1:2]
        # u = 1 - exp(lambda*x) * cos(2*pi*y)
        return 1.0 - torch.exp(self.lam * xv) * torch.cos(2 * np.pi * yv)

    def exact_grad(self, x_in):
        # Gradients for u-velocity
        xv, yv = x_in[:, 0:1], x_in[:, 1:2]
        exp_term = torch.exp(self.lam * xv)
        cos_term = torch.cos(2 * np.pi * yv)
        sin_term = torch.sin(2 * np.pi * yv)

        # du/dx = -lambda * exp(lambda*x) * cos(2*pi*y)
        du_dx = -self.lam * exp_term * cos_term

        # du/dy = -exp(lambda*x) * (-2*pi * sin(2*pi*y))
        #       = 2*pi * exp(lambda*x) * sin(2*pi*y)
        du_dy = 2 * np.pi * exp_term * sin_term

        return torch.cat([du_dx, du_dy], dim=1)

    def get_train_data(self, n_samples=5000):
        # Domain typically used for Kovasznay
        x_space = torch.rand(n_samples, 1, device=device) * 1.5 - 0.5 # [-0.5, 1.0]
        y_space = torch.rand(n_samples, 1, device=device) * 2.0 - 0.5 # [-0.5, 1.5]

        x_pde = torch.cat([x_space, y_space], dim=1)
        u_true = self.exact(x_pde)

        return x_pde, [(x_pde, u_true, 'data_fit')]

    def build(self, slv, x_pde, bcs):
        As, bs = [], []
        for (pts, vals, _) in bcs:
            h, _, _ = slv.get_features(pts)
            As.append(h)
            bs.append(vals)
        return torch.cat(As), torch.cat(bs)

    def get_test_points(self, n=2000):
        x_space = torch.rand(n, 1, device=device) * 1.5 - 0.5
        y_space = torch.rand(n, 1, device=device) * 2.0 - 0.5
        return torch.cat([x_space, y_space], dim=1)


class GrayScott_Pulse:
    """
    Gray-Scott Reaction Diffusion (Synthetic Pulse)
    Since the real GS equation [cite: 1088] requires a numerical solver
    to generate ground truth, we test the regression capability on a
    synthetic 'Pulse' pattern that mimics the sharp gradients of GS.

    Target: u(x,t) = exp( - ((x - ct)^2) / sigma )
    """
    def __init__(self):
        self.name = "Gray-Scott (Pulse)"
        self.dim = 2 # x, t
        self.c = 0.5 # speed
        self.sigma = 0.02 # Sharpness (similar to diffusivity epsilon)

    def exact(self, x_in):
        xv, tv = x_in[:, 0:1], x_in[:, 1:2]
        # Traveling Gaussian pulse
        arg = -((xv - self.c * tv)**2) / self.sigma
        return torch.exp(arg)

    def exact_grad(self, x_in):
        xv, tv = x_in[:, 0:1], x_in[:, 1:2]

        arg = -((xv - self.c * tv)**2) / self.sigma
        u = torch.exp(arg)

        # Chain rule
        # d(arg)/dx = -2(x - ct)/sigma
        darg_dx = -2 * (xv - self.c * tv) / self.sigma

        # d(arg)/dt = -2(x - ct)/sigma * (-c)
        darg_dt = -2 * (xv - self.c * tv) / self.sigma * (-self.c)

        du_dx = u * darg_dx
        du_dt = u * darg_dt

        return torch.cat([du_dx, du_dt], dim=1)

    def get_train_data(self, n_samples=5000):
        # Domain: x in [0, 1], t in [0, 1]
        x_pde = torch.rand(n_samples, 2, device=device)
        u_true = self.exact(x_pde)
        return x_pde, [(x_pde, u_true, 'data_fit')]

    def build(self, slv, x_pde, bcs):
        As, bs = [], []
        for (pts, vals, _) in bcs:
            h, _, _ = slv.get_features(pts)
            As.append(h)
            bs.append(vals)
        return torch.cat(As), torch.cat(bs)

    def get_test_points(self, n=2000):
        return torch.rand(n, 2, device=device)

class Bratu2D_Regression:
    """
    Bratu 2D (Regression Mode)
    Equation: -Delta u - lambda * e^u = f (Ignored for regression)
    Task: Fit the analytic solution u = sin(pi*x)sin(pi*y) directly.
    """
    def __init__(self):
        self.name = "Bratu 2D (Reg)"
        self.dim = 2
        # Lambda is part of the PDE, but for regression we just fit the shape
        self.lam = 1.0

    def exact(self, x):
        return torch.sin(np.pi * x[:, 0:1]) * torch.sin(np.pi * x[:, 1:2])

    def exact_grad(self, x):
        # du/dx = pi * cos(pi*x) * sin(pi*y)
        # du/dy = pi * sin(pi*x) * cos(pi*y)
        sx = torch.sin(np.pi * x[:, 0:1])
        cx = torch.cos(np.pi * x[:, 0:1])
        sy = torch.sin(np.pi * x[:, 1:2])
        cy = torch.cos(np.pi * x[:, 1:2])

        du_dx = np.pi * cx * sy
        du_dy = np.pi * sx * cy
        return torch.cat([du_dx, du_dy], dim=1)

    def get_train_data(self, n_samples=5000):
        # Domain: [0, 1] x [0, 1]
        x_pde = torch.rand(n_samples, 2, device=device)
        u_true = self.exact(x_pde)

        # Return format for Regression: x, bcs (where bcs contains the full fit data)
        return x_pde, [(x_pde, u_true, 'data_fit')]

    def build(self, slv, x_pde, bcs):
        # Standard Regression Build: A = H, b = u
        As, bs = [], []
        for (pts, vals, _) in bcs:
            h, _, _ = slv.get_features(pts)
            As.append(h)
            bs.append(vals)
        return torch.cat(As), torch.cat(bs)

    def get_test_points(self, n=5000):
        return torch.rand(n, 2, device=device)


class NLHelmholtz2D_Regression:
    """
    Nonlinear Helmholtz 2D (Regression Mode)
    Equation: Delta u + k^2 u + alpha * u^3 = f (Ignored for regression)
    Task: Fit the analytic solution u = sin(kx)sin(ky) directly.
    """
    def __init__(self):
        self.name = "NL-Helmholtz (Reg)"
        self.dim = 2
        self.k = 3.0 # Matches the non-linear solver default

    def exact(self, x):
        return torch.sin(self.k * x[:, 0:1]) * torch.sin(self.k * x[:, 1:2])

    def exact_grad(self, x):
        k = self.k
        sx = torch.sin(k * x[:, 0:1])
        cx = torch.cos(k * x[:, 0:1])
        sy = torch.sin(k * x[:, 1:2])
        cy = torch.cos(k * x[:, 1:2])

        du_dx = k * cx * sy
        du_dy = k * sx * cy
        return torch.cat([du_dx, du_dy], dim=1)

    def get_train_data(self, n_samples=5000):
        # Domain: [0, 1] x [0, 1]
        x_pde = torch.rand(n_samples, 2, device=device)
        u_true = self.exact(x_pde)
        return x_pde, [(x_pde, u_true, 'data_fit')]

    def build(self, slv, x_pde, bcs):
        # Standard Regression Build: A = H, b = u
        As, bs = [], []
        for (pts, vals, _) in bcs:
            h, _, _ = slv.get_features(pts)
            As.append(h)
            bs.append(vals)
        return torch.cat(As), torch.cat(bs)

    def get_test_points(self, n=5000):
        return torch.rand(n, 2, device=device)

# ==============================================================================
# 5. EXECUTION, COMPARISON & PLOTTING
# ==============================================================================

def run_solver_with_scale(solver_class, problem, scale, seed=42):
    """Run solver and return (Value Error, Gradient Error, Time)"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    solver = solver_class(problem.dim)

    # --- ROBUST DATA UNPACKING ---
    data = problem.get_train_data()
    if len(data) == 3:
        x_pde, bcs, f_pde = data
        build_args = (bcs, f_pde)
    else:
        x_pde, bcs = data
        build_args = (bcs,)

    # --- SCALE ADJUSTMENT ---
    # Check if problem requires anisotropic scaling (vector scale)
    if hasattr(problem, 'scale_multipliers'):
        # effective_scale will be a list [scale*1, scale*1, scale*60]
        effective_scale = [scale * m for m in problem.scale_multipliers]
    else:
        effective_scale = scale

    # Build & Solve
    start = time.time()
    for _ in range(3): # 3 blocks
        solver.add_block(hidden_size=500, scale=effective_scale)

    A, b = problem.build(solver, x_pde, *build_args)
    solver.beta = solve_lstsq(A, b)
    runtime = time.time() - start

    # --- EVALUATION ---
    torch.manual_seed(999)
    xt = problem.get_test_points(2000)

    u_true = problem.exact(xt)
    grad_true = problem.exact_grad(xt)
    u_pred, grad_pred = solver.predict_with_grad(xt)

    val_l2 = (torch.norm(u_pred - u_true) / (torch.norm(u_true) + 1e-8)).item()

    norm_grad_true = torch.norm(grad_true)
    if norm_grad_true < 1e-6: norm_grad_true = 1.0
    grad_l2 = (torch.norm(grad_pred - grad_true) / norm_grad_true).item()

    return runtime, val_l2, grad_l2

def grid_search_scale(solver_class, problem, scales, n_trials=3):
    best_scale = scales[0] # Default to first scale if all fail
    best_val_error = float('inf')
    best_grad_error = float('inf')
    best_runtime = 0.0
    results = {}

    for scale in scales:
        val_errors = []
        grad_errors = []
        runtimes = []

        for seed in range(n_trials):
            try:
                rt, v_err, g_err = run_solver_with_scale(solver_class, problem, scale, seed=seed)
                # Filter out NaNs immediately
                if np.isnan(v_err) or np.isinf(v_err):
                    v_err = 1e10 # Treat as high error
                if np.isnan(g_err) or np.isinf(g_err):
                    g_err = 1e10

                val_errors.append(v_err)
                grad_errors.append(g_err)
                runtimes.append(rt)
            except Exception as e:
                # Catch solver crashes (e.g. singular matrix)
                val_errors.append(1e10)
                grad_errors.append(1e10)
                runtimes.append(0.0)

        mean_val = np.mean(val_errors)
        mean_grad = np.mean(grad_errors)
        mean_runtime = np.mean(runtimes)

        results[scale] = (mean_val, mean_grad)

        if mean_val < best_val_error:
            best_val_error = mean_val
            best_grad_error = mean_grad
            best_scale = scale
            best_runtime = mean_runtime

    return best_scale, best_val_error, best_grad_error, best_runtime, results

def plot_and_save_sensitivity(problem_name, scales, results_rff, results_pielm):
    """
    Plots the error sensitivity to scale (bandwidth) and saves to PDF.
    This visualizes the 'Spectral Bias' of the solver.
    """
    # Extract Data for Plotting
    val_rff = [results_rff[s][0] for s in scales]
    grad_rff = [results_rff[s][1] for s in scales]

    val_pielm = [results_pielm[s][0] for s in scales]
    grad_pielm = [results_pielm[s][1] for s in scales]

    # Create Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Subplot 1: Value Error
    ax1.plot(scales, val_rff, 'b-o', label='Fast-LSQ (Sin)', linewidth=2, markersize=6)
    ax1.plot(scales, val_pielm, 'r--s', label='PIELM (Tanh)', linewidth=2, markersize=6)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Scale (Frequency Bandwidth $\\sigma$)')
    ax1.set_ylabel('Relative L2 Error (Value)')
    ax1.set_title(f'{problem_name}: Function Approx.')
    ax1.legend()
    ax1.grid(True, which="both", ls="-", alpha=0.3)

    # Subplot 2: Gradient Error (The "Physics" Error)
    ax2.plot(scales, grad_rff, 'b-o', label='Fast-LSQ (Sin)', linewidth=2, markersize=6)
    ax2.plot(scales, grad_pielm, 'r--s', label='PIELM (Tanh)', linewidth=2, markersize=6)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Scale (Frequency Bandwidth $\\sigma$)')
    ax2.set_ylabel('Relative L2 Error (Gradient)')
    ax2.set_title(f'{problem_name}: Derivative Accuracy')
    ax2.legend()
    ax2.grid(True, which="both", ls="-", alpha=0.3)

    plt.tight_layout()

    # Save to PDF
    clean_name = problem_name.replace(" ", "_").replace("/", "_")
    filename = f"Sensitivity_{clean_name}.pdf"
    plt.savefig(filename)
    plt.close() # Close figure to free memory
    print(f"   -> Saved plot to: {filename}")

def run_fair_comparison():
    problems = [PoissonND(), HeatND(), Wave1D(), Wave2D_MS(), Burgers1D_Regression(),
                KdV_Regression(), ReactionDiffusion_Regression(), Helmholtz2D(),
                SineGordon_Regression(), Maxwell2D_TM(), KleinGordon_Regression(),
                GrayScott_Pulse(), NavierStokes2D_Kovasznay(),
                Bratu2D_Regression(),NLHelmholtz2D_Regression(),]

    # Scale grid
    scales = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]

    print(f"\n{'='*115}")
    print(f"{'PROBLEM':<15} | {'METHOD':<15} | {'BEST SCALE':<10} | {'TIME (s)':<10} | {'VALUE ERROR':<12} | {'GRAD ERROR':<12}")
    print(f"{'='*115}")

    for problem in problems:
        # Fast-LSQ
        s_rff, v_rff, g_rff, t_rff, res_rff = grid_search_scale(FastLSQSolver, problem, scales)
        print(f"{problem.name:<15} | {'Fast-LSQ (sin)':<15} | {s_rff:<10.1f} | {t_rff:<10.4f} | {v_rff:.2e}     | {g_rff:.2e}")

        # PIELM
        s_pielm, v_pielm, g_pielm, t_pielm, res_pielm = grid_search_scale(PIELMSolver, problem, scales)
        print(f"{'':<15} | {'PIELM (tanh)':<15} | {s_pielm:<10.1f} | {t_pielm:<10.4f} | {v_pielm:.2e}     | {g_pielm:.2e}")

        # --- PLOTTING ---
        plot_and_save_sensitivity(problem.name, scales, res_rff, res_pielm)

        print("-" * 115)

if __name__ == "__main__":
    run_fair_comparison()
