# Import Libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import visdom
import vaex
from assimulo.solvers import ExplicitEuler
import sys
import os

# Add the parent directory of 'optimizer' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'optimizer')))

from newton_cg import NewtonCG

# PDE parameter
global gamma_1
gamma_1 = 0.005

# Define the neural network for u(t,x)
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.hidden_layers = nn.Sequential(
            nn.Linear(2, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, tx):
        return self.hidden_layers(tx)

# Compute derivatives using automatic differentiation for PDE loss
def compute_pde_loss(model, tx):
    tx = tx.clone().detach().requires_grad_(True).to(torch.float32)
    u = model(tx)

    grads = torch.autograd.grad(u, tx, torch.ones_like(u), create_graph=True)[0]
    u_t = grads[:, 0:1]
    u_x = grads[:, 1:2]

    u_xx = torch.autograd.grad(u_x, tx, torch.ones_like(u_x), create_graph=True)[0][:, 1:2]

    f = u_t - gamma_1 * u_xx + 5 * u**3 - 5 * u
    return torch.mean(f**2), f**2

# Compute derivatives using automatic differentiation for IC loss
def compute_ic_loss(model, x_ic):
    t0_x = torch.cat((torch.zeros_like(x_ic), x_ic), dim=1)
    u_pred = model(t0_x)
    u_true = x_ic**2 * torch.cos(torch.pi * x_ic)
    return torch.mean((u_pred - u_true)**2)

# Compute derivatives using automatic differentiation for BC loss
def compute_bc_loss(model, t_bc):
    x_left = -torch.ones_like(t_bc)
    x_right = torch.ones_like(t_bc)

    tx_left = torch.cat((t_bc, x_left), dim=1)
    tx_right = torch.cat((t_bc, x_right), dim=1)

    u_left = model(tx_left)
    u_right = model(tx_right)

    tx_left.requires_grad = True
    u_x_left = torch.autograd.grad(model(tx_left), tx_left, torch.ones_like(u_left), create_graph=True)[0][:, 1:2]
    
    tx_right.requires_grad = True
    u_x_right = torch.autograd.grad(model(tx_right), tx_right, torch.ones_like(u_right), create_graph=True)[0][:, 1:2]

    loss_bc = torch.mean((u_left - u_right) ** 2) + torch.mean((u_x_left - u_x_right) ** 2)
    return loss_bc

# AAS-HGR samplingof collocation points
def sample_points_exp_res_grad_hess(model, t_x_matrix, num_samples):

    Z = t_x_matrix.clone().detach().requires_grad_(True) 

    u = model(Z)                                                 
    grads_u = torch.autograd.grad(u, Z, torch.ones_like(u), create_graph=True)[0] 
    u_t, u_x = grads_u[:,0:1], grads_u[:,1:2]
    u_xx = torch.autograd.grad(u_x, Z, torch.ones_like(u_x), create_graph=True)[0][:,1:2]  
    f = (u_t - gamma_1*u_xx + 5*u**3 - 5*u).view(-1)             

    R = f.pow(2)                                                 

    grad_f = torch.autograd.grad(f, Z, torch.ones_like(f), create_graph=True)[0]
    G = grad_f.norm(dim=1)                                     

    N, d = Z.shape
    hess_sq = torch.zeros(N)
    for k in range(d):
        H_row = torch.autograd.grad(grad_f[:,k], Z,
                                    torch.ones_like(grad_f[:,k]),
                                    retain_graph=(k<d-1))[0]      
        hess_sq += H_row.pow(2).sum(dim=1)
    
    H = torch.sqrt(hess_sq + 1e-16)                             

    def normalize01(x):
        x_min, x_max = x.min(), x.max()
        return (x - x_min) / (x_max - x_min + 1e-16)

    Rn = normalize01(R)
    Gn = normalize01(G)
    Hn = normalize01(H)

    beta_R = 0.6
    beta_G = 0.3
    beta_H = 0.1

    scores = beta_R * Rn + beta_G * Gn + beta_H * Hn
    
    probs  = scores / (scores.sum() + 1e-16)

    probs_np = probs.detach().numpy()         

    idx = np.random.choice(
        len(probs_np), 
        size=num_samples, 
        replace=True, 
        p=probs_np
    )

    noise = ((torch.rand_like(t_x_matrix[idx]) - 0.5) * 0.1)/0.5 
    t_x_return = t_x_matrix[idx] + noise
    
    t_x_return[:, 0] = t_x_return[:, 0].clamp(0, 1) 
    t_x_return[:, 1] = t_x_return[:, 1].clamp(-1, 1)
    
    return t_x_return                                 


# Function to generate points within a specific time interval
def generate_points_in_time_interval(t_min, t_max, N_pde, N_bc):
    t_pde = (t_max - t_min) * torch.rand(N_pde, 1, dtype=torch.float32) + t_min
    x_pde = (2 * torch.rand(N_pde, 1, dtype=torch.float32) - 1)
    tx_pde = torch.cat((t_pde, x_pde), dim=1)
    
    t_bc = (t_max - t_min) * torch.rand(N_bc, 1, dtype=torch.float32) + t_min
    
    return tx_pde, t_bc

# Initialize model and hyperparameters
model = PINN()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Time-adaptive approach parameters
N_pde, N_ic, N_bc = 5000, 512, 200
extra_samples_per_epoch = 500
x_ic = (2 * torch.rand(N_ic, 1, dtype=torch.float32) - 1)

# Define time intervals for the time-adaptive approach
time_intervals = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
max_iterations_per_interval = 20000
loss_threshold = 1e-5

# Triangular sampling schedule

epochs = max_iterations_per_interval          
sampling_step = 500
triangle_cycle_epochs = 3000
gap_epochs = 1000
active_sampling_epochs = triangle_cycle_epochs
samples_per_peak = 600

samples_schedule = []
steps_per_cycle = active_sampling_epochs // sampling_step
total_cycles = epochs // (triangle_cycle_epochs + gap_epochs)

for cycle in range(total_cycles):
    for i in range(steps_per_cycle):
        normalized_pos = i / (steps_per_cycle - 1)
        triangle_value = 1 - abs(2 * normalized_pos - 1)
        samples_schedule.append(int(samples_per_peak * triangle_value))
    
    gap_steps = gap_epochs // sampling_step
    samples_schedule.extend([0] * gap_steps)

# Rescale to target total 40000 points
scale_factor = 40000 / sum(samples_schedule)
samples_schedule = [int(s * scale_factor) for s in samples_schedule]

print(f"Total samples scheduled: {sum(samples_schedule)}")

# Time-adaptive training
for i, t_max in enumerate(time_intervals):
    print(f"\n{'='*50}\nTraining on time interval [0, {t_max}]\n{'='*50}")
    
    # Generate points for the current time interval
    tx_pde, t_bc = generate_points_in_time_interval(0.0, t_max, N_pde, N_bc)
    
    # If not the first interval, add some points from previous intervals to maintain knowledge
    if i > 0:
        prev_tx_pde, prev_t_bc = generate_points_in_time_interval(0.0, time_intervals[i-1], N_pde//2, N_bc//2)
        tx_pde = torch.cat((tx_pde, prev_tx_pde), dim=0)
        t_bc = torch.cat((t_bc, prev_t_bc), dim=0)
    
    # Training loop with adaptive sampling for the current time interval
    best_loss = float('inf')
    for epoch in tqdm(range(max_iterations_per_interval)):
        optimizer.zero_grad()
        loss_pde, loss_vector = compute_pde_loss(model, tx_pde)
        loss_ic = compute_ic_loss(model, x_ic)
        loss_bc = compute_bc_loss(model, t_bc)
        loss = loss_pde + 100*loss_ic + loss_bc
        loss.backward()
        optimizer.step()
            

        if epoch % sampling_step == 0 and epoch != 0:
            idx = epoch // sampling_step
            if idx < len(samples_schedule):
                extra_samples = samples_schedule[idx]
                if extra_samples > 0:
                    new_tx_pde = sample_points_exp_res_grad_hess(model, tx_pde, extra_samples)
                    mask = (new_tx_pde[:, 0] <= t_max)
                    new_tx_pde = new_tx_pde[mask]
                    if len(new_tx_pde) > 0:
                        tx_pde = torch.cat((tx_pde, new_tx_pde), dim=0)
                        print(f"Epoch {epoch}: Added {extra_samples} points | Total points = {len(tx_pde)}")
            
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f} | PDE Loss = {loss_pde.item():.6f} | IC Loss = {loss_ic.item():.6f} | BC Loss = {loss_bc.item():.6f}")
            
            # Check if we've reached the threshold
            if loss.item() < loss_threshold:
                print(f"Loss threshold reached at epoch {epoch}")
                break
            
            # Track best loss for this time interval
            if loss.item() < best_loss:
                best_loss = loss.item()
    
    # LBFGS fine-tuning for each time interval
    print(f"\nStarting L-BFGS fine-tuning for time interval [0, {t_max}]...")
    optimizer_lbfgs = optim.LBFGS(model.parameters(), lr=1, max_iter=3000, 
                                 history_size=50, line_search_fn="strong_wolfe")
    
    def closure():
        optimizer_lbfgs.zero_grad()
        loss_pde, _ = compute_pde_loss(model, tx_pde)
        loss_ic = compute_ic_loss(model, x_ic)
        loss_bc = compute_bc_loss(model, t_bc)
        loss = loss_pde + 100*loss_ic + loss_bc
        loss.backward()
        print(f"L-BFGS Loss: {loss.item()}")
        return loss
    
    optimizer_lbfgs.step(closure)
    print(f"L-BFGS fine-tuning complete for time interval [0, {t_max}]")

# Final Newton-CG Fine-tuning 
optimizer_newton = NewtonCG(model.parameters(), lr=1, rank=10, mu=1e-6, cg_tol=1e-10, cg_max_iters=1000, line_search_fn="armijo")

def newton_closure():
    optimizer_newton.zero_grad()
    loss_pde, _ = compute_pde_loss(model, tx_pde)
    loss_ic = compute_ic_loss(model, x_ic)
    loss_bc = compute_bc_loss(model, t_bc)
    loss = loss_pde + 100*loss_ic + loss_bc
    loss.backward(create_graph=True) 
    grad_tuple = tuple(p.grad for p in model.parameters())
    print(f"Newton-CG Loss: {loss.item()}")
    return loss, grad_tuple

print("Starting Newton-CG fine-tuning...")

# Calling update_preconditioner once before the first step
loss, grad_tuple = newton_closure()
optimizer_newton.update_preconditioner(grad_tuple)

loss_list = []

for i in range(4): # Perform a few Newton-CG steps
    loss, grad_tuple = optimizer_newton.step(newton_closure)
    optimizer_newton.update_preconditioner(grad_tuple)
    loss_list.append(loss.item())

print("Newton-CG fine-tuning complete!")

print("\nTime-adaptive training complete!")

# Plotting results

# Computing Numerical Solution 
# Discretization parameters
Nx = 100  
Nt = 1000 
x_min, x_max = -1, 1
t_min, t_max = 0, 1

h = (x_max - x_min) / (Nx - 1) 
k = (t_max - t_min) / Nt 
x = np.linspace(x_min, x_max, Nx)
t = np.linspace(t_min, t_max, Nt)

# Stability condition check
alpha = gamma_1 * k / h**2
if alpha > 0.5:
    print(f"Warning: Stability condition may not be satisfied, α = {alpha:.6f}")

# Initialize solution matrix
u_true = np.zeros((Nx, Nt))
u_true[:, 0] = x**2 * np.cos(np.pi * x)  # Initial condition

# Time-stepping loop using explicit finite difference method
for j in range(Nt - 1):
    u_new = np.copy(u_true[:, j])

    # Apply finite difference scheme for interior points
    for i in range(1, Nx - 1):
        uxx = (u_true[i-1, j] - 2*u_true[i, j] + u_true[i+1, j]) / h**2
        nonlinear_term = -5 * (u_true[i, j]**3) + 5 * u_true[i, j]
        u_new[i] = u_true[i, j] + k * (gamma_1 * uxx + nonlinear_term)

    # Periodic boundary conditions
    u_new[0] = u_new[-2]  # u(-1) = u(1)
    u_new[-1] = u_new[1]  # ux(-1) = ux(1)

    # Update the solution
    u_true[:, j+1] = u_new

# Computing PINN Predictions
# Convert x and t into a mesh grid
X, T = np.meshgrid(x, t)
XT = np.hstack((T.flatten()[:, None], X.flatten()[:, None]))  # Shape (Nx*Nt, 2)

# Convert to torch tensor
XT_tensor = torch.tensor(XT, dtype=torch.float32)

# Get predictions from PINN
with torch.no_grad():
    u_pred = model(XT_tensor).numpy().reshape(Nt, Nx)  # Reshape back to (Nt, Nx)

# Compute error
error = np.abs(u_true.T - u_pred)  # Element-wise absolute error

# L2 Error 
L2_error = np.linalg.norm(error, ord='fro')/np.sqrt(len(error)*len(error[0]))

# Plotting Comparisons at Specific Time Steps 
time_indices = [0, int(0.25 * Nt), int(0.75 * Nt), Nt - 1]
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

for idx, ax in enumerate(axs.flatten()):
    t_idx = time_indices[idx]
    ax.plot(x, u_true[:, t_idx], 'b-', label="Numerical", linewidth=1.5)
    ax.plot(x, u_pred[t_idx, :], 'r--', label="PINN Prediction", linewidth=1.5)
    ax.set_title(f"t = {t[t_idx]:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.legend()

plt.tight_layout()
plt.show()

# Heatmap Visualization
# PINN Prediction Heatmap
plt.figure(figsize=(7, 6))
im1 = plt.imshow(u_pred, extent=[-1, 1, 0, 1], origin="lower", aspect="auto", cmap="jet")
plt.title("PINN Prediction")
plt.xlabel("x")
plt.ylabel("t")
plt.colorbar(im1)
plt.tight_layout()
plt.show()

# Numerical Solution Heatmap
plt.figure(figsize=(7, 6))
im2 = plt.imshow(u_true.T, extent=[-1, 1, 0, 1], origin="lower", aspect="auto", cmap="jet")
plt.title("Numerical Solution")
plt.xlabel("x")
plt.ylabel("t")
plt.colorbar(im2)
plt.tight_layout()
plt.show()

# Error Heatmap
plt.figure(figsize=(7, 6))
im3 = plt.imshow(error, extent=[-1, 1, 0, 1], origin="lower", aspect="auto", cmap="jet")
plt.title("Absolute Error")
plt.xlabel("x") 
plt.ylabel("t")
plt.colorbar(im3)
plt.tight_layout()
plt.show()
