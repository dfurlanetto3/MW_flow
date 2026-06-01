# -*- coding: utf-8 -*-
"""
Flow field through monitoring well equipped with flow-converging screen

Created on Sun Jul 13 11:32:02 2025

@author: Davide Furlanetto, University of Padova
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
# import pandas as pd
plt.rcParams.update({'font.size':12})

save_folder = '...'
comsol_folder = '...'
os.chdir(save_folder)

#%% Geometric and hydraulic properties

# Define MW radii [m]
R1 = 0.026 
R2 = 0.029 
R3 = 0.050

R0 = 0.2 * R1

r0 = R0 / R1
print('r_0 = ' , r0)
r1 = 1
r2 = R2 / R1
r3 = R3 / R1

# water properties at 10°C
rho_w = 999.7
mu_w = 1.32e-3 
g_const = 9.806

# set hydraulic conductivities [m/s] and convert to dimensionless permeabilities [-]
k1 = 1E-1 * mu_w / (rho_w * g_const) / R1**2
k2 = 1E-4 * mu_w / (rho_w * g_const) / R1**2
k3 = 1E-5 * mu_w / (rho_w * g_const) / R1**2

# # Directly set permeabilities [L^2]
# k1 = 1E-3 / R1**2
# k2 = 6E-7 / R1**2
# k3 = 3E-7 / R1**2

alpha_drost = 8 / (
    (1+k3/k2)*( 1 + (R1/R2)**2 + k2/k1 * (1-(R1/R2)**2)) + 
    (1-k3/k2)*( (R1/R3)**2 + (R2/R3)**2 + 
    (k2/k1) * ( (R1/R3)**2 - (R2/R3)**2) )
    )

alpha_star = 2 /  (
    (1 / r2) * (k3 / k2 + 1) *
    (0.5 * (k2 / k1 * np.log(r2) + 1) + k2)
    +
    (r2 / r3**2) * (k3 / k2 - 1) *
    (0.5 * (k2 / k1 * np.log(r2) - 1) + k2)
        )

# ------------- Help variables -------------- #
lnr0 = np.log(r0)
lnr2 = np.log(r2)

eps = (r0**2 - 1)*(lnr0*(r0**2 + 1) - r0**2 + 1)

gamma = 2*lnr0*(1 + r0**4) - r0**4 + 8*r0**2 - 7

Lambda = (1/r2 * (k3/k2 + 1) * (eps*(k2/k1*lnr2 + 1) - gamma*k2) 
        + r2/r3**2 * (k3/k2 - 1) * (eps*(k2/k1 * lnr2 - 1) - gamma*k2))
        
beta = 4 * eps / (Lambda)

print("Alpha Drost: ", alpha_drost)
print("Alpha star: ", alpha_star)
print("Beta: ", beta)

#%% Define Cartesian grid
nx_vals = 2000
ny_vals = 2000
x_vals = np.linspace(-6 , 6 , nx_vals)
y_vals = np.linspace(-5 , 5 , ny_vals)
X, Y = np.meshgrid(x_vals, y_vals)

# Convert Cartesian to polar coordinates
r_t = np.sqrt(X**2 + Y**2)
theta = np.arctan2(Y, X)

r = r_t*R1 # non-dimensional radius

#% Prepare for plots
nx, ny = 2, 34
xs = np.linspace(x_vals.min(), x_vals.max(), nx)
xs = [-5.]
ys = np.linspace(y_vals.min(), y_vals.max(), ny)
XS, YS = np.meshgrid(xs, ys)

start_points = np.column_stack([XS.ravel(), YS.ravel()])

#%% Functions

def compute_coefficients(r0, r2, r3,
                         k1, k2, k3, 
                         occlusion=True):
    """ Function to compute coefficients of the analytic solution.
    Note that no consistency checks are made on the dimensionless radii.
    
    Arguments:
    ----------
    - r0 : dimensionless radius of the occlusion (if present) [0 > r0 > 1];
            Small values of r0 cause numerical errors.
    - r2 : dimensionless outer radius of the well screen [r2 > 1];
    - r3 : dimensionless outer radius of the filter pack [r3 > r2 > 1];
    
    - k1 : dimensionless permeability of the well screen;
    - k2 : dimensionless permeability of the filter pack;
    - k3 : dimensionless permeability of the formation;
    - occlusion : [bool] If True (default) computes coefficients
                for the case with internal occlusion. If False
                computes coefficients for the case without occlusion
                (r0 can be set to any value).
                
    Returns:
    --------
    Coefficients A, B, C, D, E, F, G, H, L
    """

    if occlusion:
        
        # ------------- Help variables -------------- #
        lnr0 = np.log1p(r0 - 1)
        lnr2 = np.log(r2)

        eps = (r0**2 - 1)*(lnr0*(r0**2 + 1) - r0**2 + 1)

        gamma = 2*lnr0*(1 + r0**4) - r0**4 + 8*r0**2 - 7

        Lambda = (1/r2 * (k3/k2 + 1) * (eps*(k2/k1*lnr2 + 1) - gamma*k2) 
                + r2/r3**2 * (k3/k2 - 1) * (eps*(k2/k1 * lnr2 - 1) - gamma*k2))
        
        # -------------------------------
        # Compute coefficients
        # -------------------------------
        
        A = 1/Lambda * (2*lnr0*(r0**4 - 3) - r0**4 + 4*r0**2 - 3)
        
        B = 1/Lambda * r0**2 * (2*lnr0*(r0**2) - 3*r0**2 + 3)
        
        C = 1/Lambda * (2*lnr0 + r0**2 - 1)
        
        D = 1/Lambda * 2*(3 - r0**4 - 2*r0**2)
        
        E = - 1/(k1*Lambda) * 4 * eps
        
        F = 4 * gamma / Lambda
        
        G = - 2 / (r2*k2*Lambda) * (eps*(k2/k1*lnr2 + 1) - gamma*k2) 
        
        H = -2*r2 / (k2*Lambda) * (eps*(k2/k1 * lnr2 - 1) - gamma*k2)
        
        L = 1/(k2+k3) * (r3**2*(k2/k3-1) - 4*r2/Lambda * (eps*(k2/k1 * lnr2 - 1) 
                                                          - gamma*k2))

    else:
        
        Lambda_star = (
            (1 / r2) * (k3 / k2 + 1) *
            (0.5 * (k2 / k1 * np.log(r2) + 1) + k2)
            +
            (r2 / r3**2) * (k3 / k2 - 1) *
            (0.5 * (k2 / k1 * np.log(r2) - 1) + k2)
        )

        A = 3 / Lambda_star
        
        B = 0
        
        C = -1 / Lambda_star
        
        D = 0
        
        E = -2 / (k1 * Lambda_star)
        
        F = -4 / Lambda_star
        
        G = -2 / (r2*k2*Lambda_star) * ( 0.5 * (k2/k1 * np.log(r2) + 1) + k2 )
        
        H = -2*r2 / (k2*Lambda_star) * ( 0.5 * (k2/k1 * np.log(r2) - 1) + k2 )
        
        L = 1/(k2 + k3) * ( r3**2 * (k2/k3 - 1) - 4 * r2 / Lambda_star * (
                0.5 * (k2/k1 * np.log(r2) - 1) + k2 ) )

    return A, B, C, D, E, F, G, H, L

#%% CASE 1: impervious occlusion

A, B, C, D, E, F, G, H, L = compute_coefficients(r0, r2, r3,
                                                 k1, k2, k3,
                                                 occlusion=True)
# --------------------------- Darcy flow 1 ---------------------------------- #

# Darcy pressure
p_d1 = np.where((r > R1) & (r < R2), E*np.cos(theta)*np.log(r_t) + F*np.cos(theta), np.nan)
# Darcy flow rate
q_r1 = np.where((r > R1) & (r < R2), -k1*E * np.cos(theta)*(r_t)**(-1) , np.nan)
q_t1 = np.where((r > R1) & (r < R2), 0.0, np.nan)

# --------------------------- Darcy flow 2 ---------------------------------- #

# Darcy pressure
p_d2 = np.where((r > R2) & (r < R3), (G*r_t + H*r_t**(-1))*np.cos(theta), np.nan)
# Darcy flow rate
q_r2 = np.where((r > R2) & (r < R3), -k2*(G - H*r_t**(-2))*np.cos(theta), np.nan)
q_t2 = np.where((r > R2) & (r < R3), +k2*(G + H*r_t**(-2))*np.sin(theta), np.nan)

# --------------------------- Darcy flow 3 ---------------------------------- #

# Darcy pressure
p_d3 = np.where(r > R3, (-1/k3*r_t+L*r_t**(-1))*np.cos(theta), np.nan)
# Darcy flow rate
q_r3 = np.where(r > R3, ( 1 + k3*L*r_t**(-2))*np.cos(theta), np.nan)
q_t3 = np.where(r > R3, (-1 + k3*L*r_t**(-2))*np.sin(theta), np.nan)

# --------------------------- Stokes flow ----------------------------------- #

# Stokes pressure
p_s = np.where((r > R0) & (r < R1), (8*C*r_t - 2*D/r_t)* np.cos(theta), np.nan) # CHECK SIGN
# Stokes velocities
u_r = np.where((r > R0) & (r < R1), (A + B/r_t**2 + C*r_t**2 + D*np.log(r_t)) * np.cos(theta), np.nan)
u_t = np.where((r > R0) & (r < R1),-(A - B/r_t**2+3*C*r_t**2 + D*(np.log(r_t)+1))* np.sin(theta), np.nan)


# Combine Darcy zones (1, 2, 3)

p_d = np.full_like(r, np.nan)
q_r = np.full_like(r, np.nan)
q_t = np.full_like(r, np.nan)

# Zone 1
mask1 = (r > R1) & (r < R2)
p_d[mask1] = p_d1[mask1]
q_r[mask1] = q_r1[mask1]
q_t[mask1] = q_t1[mask1]

# Zone 2
mask2 = (r > R2) & (r < R3)
p_d[mask2] = p_d2[mask2]
q_r[mask2] = q_r2[mask2]
q_t[mask2] = q_t2[mask2]

# Zone 3
mask3 = r > R3
p_d[mask3] = p_d3[mask3]
q_r[mask3] = q_r3[mask3]
q_t[mask3] = q_t3[mask3]

# Convert Darcy velocities (all zones) to Cartesian
q_x = q_r * np.cos(theta) - q_t * np.sin(theta)
q_y = q_r * np.sin(theta) + q_t * np.cos(theta)

u_x = u_r * np.cos(theta) - u_t * np.sin(theta)
u_y = u_r * np.sin(theta) + u_t * np.cos(theta)

Q = np.sqrt(q_x**2+q_y**2)
U = np.sqrt(u_x**2+u_y**2)

#%% Plot: velocity magnitude field
u_plot = np.nansum([Q, U], axis=0)
u_plot = np.where(r>R0, u_plot, np.nan)
vmin = np.nanmin(u_plot)
vmax = np.nanmax(u_plot)
vel_label_U = r'$|\mathbf{u}|/q_{\infty},\quad |\mathbf{q}|/q_{\infty}$'

fig, ax = plt.subplots(figsize=(9, 4.5))
contourf_d = ax.contourf(X, Y, u_plot, levels=20,
                          cmap='viridis')

cbar = fig.colorbar(contourf_d, ax=ax, format='%.2f', fraction=0.046, pad=0.04)
cbar.set_label(vel_label_U)

# plot streamlines
ax.streamplot(X, Y, np.nansum([u_x, q_x], axis=0), np.nansum([u_y, q_y], axis=0),
               color='black', linewidth=0.7, start_points=start_points,
               broken_streamlines=False,# density=[5, 5]
                )
    
for R in [1, R2/R1, R3/R1]: #[1, R2/R1]: #
    plt.gca().add_artist(
        plt.Circle((0, 0), R, color='white', linestyle='--', fill=False,
                   linewidth=1.1))

plt.gca().add_artist(
    plt.Circle((0, 0), R0/R1, facecolor='gray', linestyle='-', fill=True,
               edgecolor='white'))

ax.set_aspect('equal', adjustable='box')
ax.set_xlim([-4, 4])
ax.set_ylim([-3, 3.])
ax.set_xlabel('$X / R_1$')
ax.set_ylabel('$Y / R_1$')

textstr = (
    rf'$\kappa_1 = {k1:.1e}$' '\n'
    rf'$\kappa_2 = {k2:.1e}$' '\n'
    rf'$\kappa_3 = {k3:.1e}$')

ax.text(
    0.725, 0.2, textstr,
    transform=ax.transAxes,
    fontsize=12,
    va='top',
    bbox=dict(boxstyle='square', facecolor='white', alpha=1))

plt.grid(False)

# plt.savefig('f102.jpg', dpi=360, format='jpg', bbox_inches='tight')
plt.show()
#%% Plot the pressures
p_plot = np.nansum([p_d, p_s], axis=0)
vmin = np.nanmin(p_plot)
vmax = np.nanmax(p_plot)

fig, ax = plt.subplots(figsize=(9, 5))
contourf_d = ax.contourf(X, Y, p_plot, levels=20,
                          cmap='viridis')

cbar = plt.colorbar(contourf_d)
cbar.set_label(r'Non-dimensional pressure')

# plot streamlines
ax.streamplot(X, Y, np.nansum([u_x, q_x], axis=0), np.nansum([u_y, q_y], axis=0),
               color='black', linewidth=0.7, start_points=start_points,
               broken_streamlines=False,# density=[5, 5]
                )
    
for R in [1, R2/R1, R3/R1]: # [1, R2/R1]: #
    plt.gca().add_artist(
        plt.Circle((0, 0), R, color='white', linestyle='--', fill=False,
                   linewidth=1.1) )
    
plt.gca().add_artist(
    plt.Circle((0, 0), R0/R1, facecolor='gray', linestyle='-', fill=True,
               edgecolor='white'))
ax.set_aspect('equal', adjustable='box')
ax.set_xlim([-4, 4])
ax.set_ylim([-3, 3.])
ax.set_xlabel('$X / R_1$')
ax.set_ylabel('$Y / R_1$')
plt.grid(False)
plt.show()
#%% Plot graphs

# num_X = pd.read_csv(os.path.join(comsol_folder,'p102_X.csv'), comment='%', header=None)
# num_Y = pd.read_csv(os.path.join(comsol_folder,'p102_Y.csv'), comment='%', header=None)

vel_label = r'$u_x/q_{\infty},\quad q_x/q_{\infty}$'

fig, axs = plt.subplots(1,2, figsize=(9, 4))
ax=axs[0]

for R in [1, R2/R1, R3/R1]: #[1, R2/R1]:#
    ax.axvline( R, linestyle='--', color='darkgray', linewidth=1.2)
    ax.axvline(-R, linestyle='--', color='darkgray', linewidth=1.2)

ax.axhline(1, color='black', linewidth=1.0)
ax.axvline(r0, color='darkgray', linewidth=1.2)

ax.plot(x_vals, u_x[int(ny_vals/2), :],linewidth=1.5)
ax.plot(x_vals, q_x[int(ny_vals/2), :],linewidth=1.5)

# add numerical for comparison
# ax.scatter(num_X[0]/R1, num_X[2]/q_inf, s=30,
#            marker='o',
#            facecolors='none',
#            edgecolors='black',
#            linewidths=.7)

# ax.scatter(num_X[0]/R1, num_X[3]/q_inf, s=30,
#            marker='o',
#            facecolors='none',
#            edgecolors='black',
#            linewidths=.7)

ax.set_xlim([0, 5])
ax.set_xlabel('$x / R_1$', fontsize=12)
ax.set_ylabel(vel_label, fontsize=12)

ax=axs[1]

for R in [1, R2/R1, R3/R1]: #[1, R2/R1]:#
    ax.axvline( R, linestyle='--', color='darkgray', linewidth=1.2)
    ax.axvline(-R, linestyle='--', color='darkgray', linewidth=1.2)

ax.axhline(1, color='black', linewidth=1.0)
ax.axvline(r0, color='darkgray', linewidth=1.2)

ax.plot(y_vals, u_x[:, int(nx_vals/2)], linewidth=1.5)
ax.plot(y_vals, q_x[:, int(nx_vals/2)], linewidth=1.5)
# add numerical for comparison
# ax.scatter(num_Y[1]/R1, num_Y[2]/q_inf,
#            marker='o',
#            s=30,
#            facecolors='none',
#            edgecolors='black',
#            linewidths=.7)

# ax.scatter(num_Y[1]/R1, num_Y[3]/q_inf,
#            s=30,
#            marker='o',
#            facecolors='none',
#            edgecolors='black',
#            linewidths=.7)

ax.set_xlim([0, 5])
ax.set_xlabel('$y / R_1$', fontsize=12)

textstr = (
    rf'$\kappa_1 = {k1:.1e}$' '\n'
    rf'$\kappa_2 = {k2:.1e}$' '\n'
    rf'$\kappa_3 = {k3:.1e}$'
)

for ax in axs:
    ax.yaxis.set_major_locator(MultipleLocator(.5))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.text(
        0.58, 0.75, textstr,
        transform=ax.transAxes,
        fontsize=12,
        va='top',
        bbox=dict(boxstyle='square', facecolor='white', alpha=1.0)
    )
    ax.set_box_aspect(1)
    
plt.grid(False)
# plt.savefig('p102.jpg', dpi=360, format='jpg', bbox_inches='tight')
plt.show()

#%% CASE 2: open borehole space

A1, B1, C1, D1, E1, F1, G1, H1, L1 = compute_coefficients(r0, r2, r3,
                                                          k1, k2, k3,
                                                          occlusion=False)
# --------------------------- Darcy flow 1 ---------------------------------- #

# Darcy pressure
p_d1 = np.where((r > R1) & (r < R2), E1*np.cos(theta)*np.log(r_t) + F1*np.cos(theta), np.nan)
# Darcy flow rate
q_r1 = np.where((r > R1) & (r < R2), -k1*E1 * np.cos(theta)*(r_t)**(-1) , np.nan)
q_t1 = np.where((r > R1) & (r < R2), 0.0, np.nan)

# --------------------------- Darcy flow 2 ---------------------------------- #

# Darcy pressure
p_d2 = np.where((r > R2) & (r < R3), (G1*r_t + H1*r_t**(-1))*np.cos(theta), np.nan)
# Darcy flow rate
q_r2 = np.where((r > R2) & (r < R3), -k2*(G1 - H1*r_t**(-2))*np.cos(theta), np.nan)
q_t2 = np.where((r > R2) & (r < R3), +k2*(G1 + H1*r_t**(-2))*np.sin(theta), np.nan)

# --------------------------- Darcy flow 3 ---------------------------------- #

# Darcy pressure
p_d3 = np.where(r > R3, (-1/k3*r_t+L1*r_t**(-1))*np.cos(theta), np.nan)
# Darcy flow rate
q_r3 = np.where(r > R3, (1 + k3*L1*r_t**(-2))*np.cos(theta), np.nan)
q_t3 = np.where(r > R3, (-1 + k3*L1*r_t**(-2))*np.sin(theta), np.nan)

# --------------------------- Stokes flow ----------------------------------- #

# Stokes pressure
p_s = np.where(r <= R1, 8*C1 * r_t * np.cos(theta), np.nan)
# Stokes velocities
u_r = np.where(r <= R1, (A1 + C1 * r_t**2) * np.cos(theta), np.nan)
u_t = np.where(r <= R1, (-A1 - 3 * C1 * r_t**2) * np.sin(theta), np.nan)

# --------------------------------------------------------------------------- #
# Combine Darcy zones (1, 2, 3)

p_d = np.full_like(r, np.nan)
q_r = np.full_like(r, np.nan)
q_t = np.full_like(r, np.nan)

# Zone 1
mask1 = (r > R1) & (r < R2)
p_d[mask1] = p_d1[mask1]
q_r[mask1] = q_r1[mask1]
q_t[mask1] = q_t1[mask1]

# Zone 2
mask2 = (r > R2) & (r < R3)
p_d[mask2] = p_d2[mask2]
q_r[mask2] = q_r2[mask2]
q_t[mask2] = q_t2[mask2]

# Zone 3
mask3 = r > R3
p_d[mask3] = p_d3[mask3]
q_r[mask3] = q_r3[mask3]
q_t[mask3] = q_t3[mask3]

# Convert Darcy velocities (all zones) to Cartesian
q_x = q_r * np.cos(theta) - q_t * np.sin(theta)
q_y = q_r * np.sin(theta) + q_t * np.cos(theta)

u_x = u_r * np.cos(theta) - u_t * np.sin(theta)
u_y = u_r * np.sin(theta) + u_t * np.cos(theta)

Q = np.sqrt(q_x**2+q_y**2)
U = np.sqrt(u_x**2+u_y**2)

#%% Plot: velocity magnitude
u_plot = np.nansum([Q, U], axis=0)
vmin = np.nanmin(u_plot)
vmax = np.nanmax(u_plot)
vel_label_U = r'$|\mathbf{u}|/q_{\infty},\quad |\mathbf{q}|/q_{\infty}$'

fig, ax = plt.subplots(figsize=(9, 4.5))
contourf_d = ax.contourf(X, Y, u_plot, levels=20,
                          cmap='viridis')

cbar = fig.colorbar(contourf_d, ax=ax, format='%.2f', fraction=0.046, pad=0.04)
cbar.set_label(vel_label_U)

# plot streamlines
ax.streamplot(X, Y, np.nansum([u_x, q_x], axis=0), np.nansum([u_y, q_y], axis=0),
               color='black', linewidth=0.7, start_points=start_points,
               broken_streamlines=False,# density=[5, 5]
                )
    
for R in [1, R2/R1, R3/R1]: # [1, R2/R1]: #
    plt.gca().add_artist(
        plt.Circle((0, 0), R, color='white', linestyle='--', fill=False,
                   linewidth=1.1)
    )

ax.set_aspect('equal', adjustable='box')
ax.set_xlim([-4, 4])
ax.set_ylim([-3, 3.])
ax.set_xlabel('$X / R_1$')
ax.set_ylabel('$Y / R_1$')

textstr = (
    rf'$\kappa_1 = {k1:.1e}$' '\n'
    rf'$\kappa_2 = {k2:.1e}$' '\n'
    rf'$\kappa_3 = {k3:.1e}$'
)

ax.text(
    0.725, 0.2, textstr,
    transform=ax.transAxes,
    fontsize=12,
    va='top',
    bbox=dict(boxstyle='square', facecolor='white', alpha=1)
)
plt.grid(False)

# plt.savefig('f402.jpg', dpi=360, format='jpg', bbox_inches='tight')
plt.show()

#%% Plot graphs

# num_X = pd.read_csv(os.path.join(comsol_folder,'p402_X.csv'), comment='%', header=None)
# num_Y = pd.read_csv(os.path.join(comsol_folder,'p402_Y.csv'), comment='%', header=None)

vel_label = r'$u_x/q_{\infty},\quad q_x/q_{\infty}$'

fig, axs = plt.subplots(1,2, figsize=(9, 4))
ax=axs[0]

for R in [1, R2/R1, R3/R1]:# [1, R2/R1]: #
    ax.axvline( R, linestyle='--', color='darkgray', linewidth=1.2)
    ax.axvline(-R, linestyle='--', color='darkgray', linewidth=1.2)

ax.axhline(1, color='black', linewidth=1.0)
# plot analytical
ax.plot(x_vals, u_x[int(ny_vals/2), :],linewidth=1.5)
ax.plot(x_vals, q_x[int(ny_vals/2), :],linewidth=1.5)
# add numerical for comparison
# ax.scatter(num_X[0]/R1, num_X[2]/q_inf, s=30,
#            marker='o',
#            facecolors='none',
#            edgecolors='black',
#            linewidths=.7)

# ax.scatter(num_X[0]/R1, num_X[3]/q_inf, s=30,
#            marker='o',
#            facecolors='none',
#            edgecolors='black',
#            linewidths=.7)

ax.set_xlim([0, 5])
ax.set_xlabel('$x / R_1$')
ax.set_ylabel(vel_label)

ax=axs[1]

for R in [1, R2/R1, R3/R1]:# [1, R2/R1]:# 
    ax.axvline( R, linestyle='--', color='darkgray', linewidth=1.2)
    ax.axvline(-R, linestyle='--', color='darkgray', linewidth=1.2)

ax.axhline(1, color='black', linewidth=1.0)

ax.plot(y_vals, u_x[:, int(nx_vals/2)], linewidth=1.5)
ax.plot(y_vals, q_x[:, int(nx_vals/2)], linewidth=1.5)

# add numerical for comparison
# ax.scatter(num_Y[1]/R1, num_Y[2]/q_inf,
#            marker='o',
#            s=30,
#            facecolors='none',
#            edgecolors='black',
#            linewidths=.7)

# ax.scatter(num_Y[1]/R1, num_Y[3]/q_inf,
#            s=30,
#            marker='o',
#            facecolors='none',
#            edgecolors='black',
#            linewidths=.7)

ax.set_xlim([0, 5])
ax.set_xlabel('$y / R_1$')
# ax.set_ylabel(vel_label)

textstr = (
    rf'$\kappa_1 = {k1:.1e}$' '\n'
    rf'$\kappa_2 = {k2:.1e}$' '\n'
    rf'$\kappa_3 = {k3:.1e}$'
)

for ax in axs:
    ax.yaxis.set_major_locator(MultipleLocator(.5))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.text(
        0.58, 0.50, textstr,
        transform=ax.transAxes,
        fontsize=12,
        va='top',
        bbox=dict(boxstyle='square', facecolor='white', alpha=1.0)
    )
    
plt.grid(False)

# plt.savefig('p402.jpg', dpi=360, format='jpg', bbox_inches='tight')
plt.show()

#%% Plot the pressures
p_plot = np.nansum([p_d, p_s], axis=0)
vmin = np.nanmin(p_plot)
vmax = np.nanmax(p_plot)

fig, ax = plt.subplots(figsize=(9, 5))
contourf_d = ax.contourf(X, Y, p_plot, levels=20,
                          cmap='viridis')

cbar = plt.colorbar(contourf_d)
cbar.set_label(r'Non-dimensional pressure')

# plot streamlines
ax.streamplot(X, Y, np.nansum([u_x, q_x], axis=0), np.nansum([u_y, q_y], axis=0),
               color='black', linewidth=0.7, start_points=start_points,
               broken_streamlines=False,# density=[5, 5]
                )
    
for R in [1, R2/R1, R3/R1]: # [1, R2/R1]: #
    plt.gca().add_artist(
        plt.Circle((0, 0), R, color='white', linestyle='--', fill=False,
                   linewidth=1.1)
    )

ax.set_aspect('equal', adjustable='box')
ax.set_xlim([-4, 4])
ax.set_ylim([-3, 3.])
ax.set_xlabel('$X / R_1$')
ax.set_ylabel('$Y / R_1$')
plt.grid(False)
plt.show()

#%% Plot x velocities components

vmin = np.min([np.nanmin(u_x), np.nanmin(q_x)])
vmax = np.max([np.nanmax(u_x), np.nanmax(q_x)])

fig, ax = plt.subplots(figsize=(9, 5))
contourf_d = ax.contourf(X, Y, u_x, levels=20,
                          cmap='viridis', vmin=vmin, vmax=vmax)
contourf_s = ax.contourf(X, Y, q_x, levels=20,
                          cmap='viridis', vmin=vmin, vmax=vmax)

cbar = plt.colorbar(contourf_s)
cbar.set_label(r'Dimensionless velocity component')

# plot streamlines
ax.streamplot(X, Y, np.nansum([u_x, q_x], axis=0), np.nansum([u_y, q_y], axis=0),
               color='black', linewidth=0.8, 
               start_points=start_points, broken_streamlines=False)

for R in [1, R2/R1, R3/R1]:
    plt.gca().add_artist(
        plt.Circle((0, 0), R, color='white', linestyle='--', fill=False))
    
# plt.gca().add_artist(
#     plt.Circle((0, 0), R0/R1, facecolor='gray', linestyle='-', fill=True,
#                edgecolor='white'))

ax.set_aspect('equal', adjustable='box')
ax.set_xlim([-4, 4])
ax.set_ylim([-3, 3])
ax.set_xlabel('$X / R_1$')
ax.set_ylabel('$Y / R_1$')
plt.grid(False)
plt.show()

#%% Plot y velocities components

vmin = np.min([np.nanmin(u_y), np.nanmin(q_y)])
vmax = np.max([np.nanmax(u_y), np.nanmax(q_y)])

fig, ax = plt.subplots(figsize=(9, 5))
contourf_d = ax.contourf(X, Y, u_y, levels=20,
                          cmap='viridis', vmin=vmin, vmax=vmax)
contourf_s = ax.contourf(X, Y, q_y, levels=20,
                          cmap='viridis', vmin=vmin, vmax=vmax)

cbar = plt.colorbar(contourf_s)
cbar.set_label(r'Dimensionless velocity component')

# plot streamlines
ax.streamplot(X, Y, np.nansum([u_x, q_x], axis=0), np.nansum([u_y, q_y], axis=0),
               color='black', linewidth=0.8, 
               start_points=start_points, broken_streamlines=False)

for R in [1, R2/R1, R3/R1]:
    plt.gca().add_artist(
        plt.Circle((0, 0), R, color='white', linestyle='--', fill=False))

# plt.gca().add_artist(
#     plt.Circle((0, 0), R0/R1, facecolor='gray', linestyle='-', fill=True,
#                edgecolor='white'))
    
ax.set_aspect('equal', adjustable='box')
ax.set_xlim([-4, 4])
ax.set_ylim([-3, 3])
ax.set_xlabel('$X / R_1$')
ax.set_ylabel('$Y / R_1$')
plt.grid(False)
plt.show()

#%% Sensitivity of beta
# Fix dimensionless radii
r1 = 1
r2 = 1.115 
r3 = 1.923

cmap = matplotlib.colormaps['magma']
allcolors = cmap(np.linspace(0, 1, 5))
colors = [allcolors[i] for i in range(4)]
linestyles = ['-', '--',  ':']         # k3/k2

# Functions
def epsilon(r0):
    return (r0**2 - 1) * (np.log(r0) * (r0**2 + 1) - r0**2 + 1)

def Lambda(r0, k1, k2, k3):
    eps = epsilon(r0)
    gam = 2*np.log(r0)*(1 + r0**4) - r0**4 + 8*r0**2 - 7

    term1 = (1/r2) * (k3/k2 + 1) * (
        eps * ((k2/k1)*np.log(r2) + 1) - gam * k2
    )

    term2 = (r2/(r3**2)) * (k3/k2 - 1) * (
        eps * ((k2/k1)*np.log(r2) - 1) - gam * k2
    )

    return term1 + term2

def beta(r0, k1, k2, k3):
    return 4 * epsilon(r0) / (Lambda(r0, k1, k2, k3) )

def betaQ(r0, k1, k2, k3):
    return beta(r0, k1, k2, k3) / (1 - r0)
#%

r0_vals = np.linspace(0.0001, 0.99999, 5000)

k2_values = [1e-4, 1e-6, 1e-8]   
ratios_21 = [10, 1, 0.1]
ratios_32 = [10, 1, 0.1, 0.01]

# %
fig, axes = plt.subplots(2, len(k2_values), figsize=(12, 8), sharex=True, sharey='row')

for col, k2 in enumerate(k2_values):

    # --- TOP ROW: beta ---
    ax = axes[0, col]

    for i, r21 in enumerate(ratios_21):
        k1 = k2 / r21

        for j, r32 in enumerate(ratios_32):
            k3 = r32 * k2

            beta_vals = beta(r0_vals, k1, k2, k3)

            ax.plot(
                r0_vals,
                beta_vals,
                color=colors[j],
                linestyle=linestyles[i],
                linewidth=1.8
            )
    
    ax.text(
        0.37, 0.95, rf"$\kappa_2 = {k2:.0e}$",
        transform=ax.transAxes,
        fontsize=12,
        va='top',
        bbox=dict(boxstyle='square', facecolor='white', alpha=1,
                  linewidth=.5))
    
    ax.set_ylim([-.1, 4.0])
    ax.set_box_aspect(1)

    if col == 0:
        ax.set_ylabel(r"$\beta$")


    # --- BOTTOM ROW: betaQ ---
    ax = axes[1, col]

    for i, r21 in enumerate(ratios_21):
        k1 = k2 / r21

        for j, r32 in enumerate(ratios_32):
            k3 = r32 * k2

            Qbeta_vals = betaQ(r0_vals, k1, k2, k3)

            ax.plot(
                r0_vals,
                np.log10(Qbeta_vals),
                color=colors[j],
                linestyle=linestyles[i],
                linewidth=1.8
            )

    ax.set_xlabel(r"$r_0$")
    ax.set_ylim([-2, 2.7])
    ax.set_box_aspect(1)
    
    ax.text(
        0.35, 0.95, rf"$\kappa_2 = {k2:.0e}$",
        transform=ax.transAxes,
        fontsize=12,
        va='top',
        bbox=dict(boxstyle='square', facecolor='white', alpha=1,
                  linewidth=.5))
    
    if col == 0:
        ax.set_ylabel(r"$\log_{10}(\beta/(1-r_0))$")


# --- Legends (shared for whole figure) ---
color_handles = [
    Line2D([0], [0], color='gray', linestyle=linestyles[j], lw=2, label=f"{ratios_21[j]}")
    for j in range(len(ratios_21))
]

style_handles = [
    Line2D([0], [0], lw=2, color=colors[i], label=f"{ratios_32[i]}")
    for i in range(len(ratios_32))
]

fig.legend(
    handles=color_handles,
    title=r"$\kappa_2/\kappa_1$",
    ncol=3,
    fontsize=12,
    bbox_to_anchor=(0.45, -0.008),
    fancybox=False,
)

fig.legend(
    handles=style_handles,
    title=r"$\kappa_3/\kappa_2$",
    ncol=4,
    fontsize=12,
    bbox_to_anchor=(0.84, -0.008),
    fancybox=False,
)

plt.tight_layout()
# plt.savefig('beta.jpg', dpi=360, format='jpg', bbox_inches='tight')
plt.show()

#%% Velocity distortion zone

r0 = .2
r1 = 1
r2 = 1.115 
r3 = 1.2

chi = np.linspace(0.001, 0.2, 1000)

k2_values = [1e-4, 1e-6, 1e-8]
ratios_21 = [10, 1, 0.01]
ratios_32 = [10, 1, 0.1, 0.01]


# %
fig, axs = plt.subplots(1,2, figsize=(9, 4), sharey=True)

for _, k2 in enumerate(k2_values):

    for i, r21 in enumerate(ratios_21):
        k1 = k2 / r21

        for j, r32 in enumerate(ratios_32):
            k3 = r32 * k2
            
            r3 = 1.2
            A3, B3, C3, D3, E3, F3, G3, H3, L3 = compute_coefficients(
                                                r0, r2, r3, k1, k2, k3,
                                                occlusion=False)
            
            r_star = np.sqrt(k3 * np.abs(L3) / chi)
            
            axs[0].plot( r_star, chi, 
                    # color='gray',
                    linewidth = 1.8,
                    color=colors[j],
                    linestyle=linestyles[i],
                    # linewidth=2,
                    alpha=0.5
                )

            r3 = 2.5
            A3, B3, C3, D3, E3, F3, G3, H3, L3 = compute_coefficients(
                                                r0, r2, r3, k1, k2, k3,
                                                occlusion=False)
            
            r_star = np.sqrt(k3 * np.abs(L3) / chi)
            
            axs[1].plot( r_star, chi, 
                    # color='gray',
                    # linewidth = 1,
                    color=colors[j],
                    linestyle=linestyles[i],
                    linewidth=1.8,
                    alpha=1
                )
for ax in axs:
    ax.axvline(10, linestyle='--', color='gray')
    ax.set_xlim([1.2, 32])
    ax.set_ylim([0.0, 0.08])
    ax.set_xlabel('$r^*/R_1$')

axs[0].axvline(12, linestyle='--', color='tab:blue')
axs[1].axvline(25, linestyle='--', color='tab:blue')
axs[0].text(
        13, 0.06, '$10 R_3$',
        color='tab:blue',
        fontsize=12,
        va='top',
        # bbox=dict(boxstyle='square', facecolor='white', alpha=1.0)
     )
axs[1].text(
        26, 0.06, '$10 R_3$',
        color='tab:blue',
        fontsize=12,
        va='top',
        # bbox=dict(boxstyle='square', facecolor='white', alpha=1.0)
     )
axs[0].text(
        5, 0.07, '$10 R_1$',
        color='darkgray',
        fontsize=12,
        va='top',
        # bbox=dict(boxstyle='square', facecolor='white', alpha=1.0)
     )
axs[1].text(
        11, 0.07, '$10 R_1$',
        color='darkgray',
        fontsize=12,
        va='top',
        # bbox=dict(boxstyle='square', facecolor='white', alpha=1.0)
     )
axs[0].set_ylabel('$\\xi$')

# plt.tight_layout()
# plt.savefig('xi.jpg', dpi=360, format='jpg', bbox_inches='tight')
plt.show()
