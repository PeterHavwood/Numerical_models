import numpy as np
import matplotlib.pyplot as plt

'''
zeta: vertical verticity
psi: stream function
'''

def init2d(params):
    Nx = params['Nx']
    init_type = params['init_type']
    
    zeta = np.zeros((Nx, Nx))

    if init_type == 1:
        # Initial condition 1: background isotropic low vorticity with central high vorticity
        zeta_bg = 1e-5
        zeta_central = 1e-4
        central_ratio = 0.1
        zeta.fill(zeta_bg)
        for i in range(Nx):
            for j in range(Nx):
                if (i - Nx//2)**2 + (j - Nx//2)**2 <= (Nx*central_ratio)**2:
                    zeta[i, j] = zeta_central + np.random.normal(0, 1e-5)

    return zeta

def solver_psi(zeta, Nx, dx, laplace_tol):
    psi = np.zeros((Nx, Nx))
    psi_new = np.zeros((Nx, Nx))

    while True:
        # inner points
        psi_new[1:-1, 1:-1] = 0.25 * (psi[:-2, 1:-1] + psi[2:, 1:-1] + psi[1:-1, :-2] + psi[1:-1, 2:] - dx**2 * zeta[1:-1, 1:-1])
        # boundary points
        psi_new[0, 1:-1] = 0.25 * (psi[-1, 1:-1] + psi[1, 1:-1] + psi[0, :-2] + psi[0, 2:] - dx**2 * zeta[0, 1:-1])
        psi_new[-1, 1:-1] = 0.25 * (psi[-2, 1:-1] + psi[0, 1:-1] + psi[-1, :-2] + psi[-1, 2:] - dx**2 * zeta[-1, 1:-1])
        psi_new[1:-1, 0] = 0.25 * (psi[:-2, 0] + psi[2:, 0] + psi[1:-1, -1] + psi[1:-1, 1] - dx**2 * zeta[1:-1, 0])
        psi_new[1:-1, -1] = 0.25 * (psi[:-2, -1] + psi[2:, -1] + psi[1:-1, -2] + psi[1:-1, 0] - dx**2 * zeta[1:-1, -1])
        if np.max(np.abs(psi_new - psi)) < laplace_tol:
            break
        psi = psi_new
        
    return psi

def get_velocity(psi, Nx, dx):
    u = np.zeros((Nx, Nx))
    v = np.zeros((Nx, Nx))

    # inner points
    u[1:-1, 1:-1] = (psi[1:-1, 2:] - psi[1:-1, :-2]) / (2 * dx)
    v[1:-1, 1:-1] = (psi[:-2, 1:-1] - psi[2:, 1:-1]) / (2 * dx)

    # boundary points
    u[0, 1:-1] = (psi[0, 2:] - psi[0, :-2]) / (2 * dx)
    u[-1, 1:-1] = (psi[-1, 2:] - psi[-1, :-2]) / (2 * dx)
    u[1:-1, 0] = (psi[:-2, 0] - psi[2:, 0]) / (2 * dx)
    u[1:-1, -1] = (psi[:-2, -1] - psi[2:, -1]) / (2 * dx)
    v[0, 1:-1] = (psi[:-2, 0] - psi[2:, 0]) / (2 * dx)
    v[-1, 1:-1] = (psi[0, 1:-1] - psi[-2, 1:-1]) / (2 * dx)
    v[1:-1, 0] = (psi[1:-1, -1] - psi[1:-1, -2]) / (2 * dx)
    v[1:-1, -1] = (psi[1:-1, 0] - psi[1:-1, -2]) / (2 * dx)

    return u, v

def advect_zeta(zeta, u, v, Nx, dx, dt):
    zeta_new = np.zeros((Nx, Nx))

    # inner points
    zeta_new[1:-1, 1:-1] = zeta[1:-1, 1:-1] - \
        dt * (u[1:-1, 1:-1] * (zeta[1:-1, 2:] - zeta[1:-1, :-2]) / (2 * dx) + \
            v[1:-1, 1:-1] * (zeta[:-2, 1:-1] - zeta[2:, 1:-1]) / (2 * dx))
    
    # boundary points
    zeta_new[0, 1:-1] = zeta[0, 1:-1] - \
        dt * (u[0, 1:-1] * (zeta[0, 2:] - zeta[0, :-2]) / (2 * dx) + \
            v[0, 1:-1] * (zeta[-1, 1:-1] - zeta[1, 1:-1]) / (2 * dx))
    zeta_new[-1, 1:-1] = zeta[-1, 1:-1] - \
        dt * (u[-1, 1:-1] * (zeta[-1, 2:] - zeta[-1, :-2]) / (2 * dx) + \
            v[-1, 1:-1] * (zeta[-2, 1:-1] - zeta[0, 1:-1]) / (2 * dx))
    zeta_new[1:-1, 0] = zeta[1:-1, 0] - \
        dt * (u[1:-1, 0] * (zeta[1:-1, 1] - zeta[1:-1, -1]) / (2 * dx) + \
            v[1:-1, 0] * (zeta[:-2, 0] - zeta[2:, 0]) / (2 * dx))
    zeta_new[1:-1, -1] = zeta[1:-1, -1] - \
        dt * (u[1:-1, -1] * (zeta[1:-1, 0] - zeta[1:-1, -2]) / (2 * dx) + \
            v[1:-1, -1] * (zeta[:-2, -1] - zeta[2:, -1]) / (2 * dx))

    return zeta_new

def model_run(zeta, params):
    dx = params['dx']
    Nx = params['Nx']
    dt = params['dt']
    Nt = params['Nt']
    laplace_tol = params['laplace_tol']

    for t in range(Nt):
        psi = solver_psi(zeta, Nx, dx, laplace_tol)
        u,v = get_velocity(psi, Nx, dx)
        zeta = advect_zeta(zeta, u, v, Nx, dx, dt)
        visualize(zeta, u,v, t)

def visualize(zeta, u,v, nt):
    plt.figure()
    plt.imshow(zeta, cmap='jet')
    plt.colorbar()
    plt.savefig(f"figures/{nt}.jpg")

    plt.figure()
    plt.quiver(u[::20, ::20], v[::20, ::20])
    plt.savefig(f"figures/velocity_{nt}.jpg")

def main():
    params = {
        'dx': 1e3,
        'Nx': 1000,
        'dt': 10,
        'Nt': 1000,
        'init_type': 1,
        'laplace_tol': 1e-6
    }
    
    zeta = init2d(params)

    model_run(zeta, params)

if __name__ == '__main__':
    main()