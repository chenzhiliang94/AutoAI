import numpy as np
from sklearn.metrics import mean_squared_error

def HeatMapLossFunction(X_local, y_local, X_global, z_global, f, g, plt):
    '''
    Plot 2D Heatmap of 3 loss functions:
    local loss, global loss, pertubed global loss

    Assumes a composite function f.g only
    '''
    grid_size = 50
    theta_0_range, theta_1_range = np.meshgrid(np.linspace(-2, 2, grid_size), np.linspace(0, 2, grid_size))
    local_loss = []
    global_loss = []
    global_pertube_loss = []
    for theta_0, theta_1 in zip(theta_0_range.flatten(), theta_1_range.flatten()):
        y_pred = g(X_local, params=[theta_0, theta_1])
        z_pred = f(y_pred, noisy = False)
        z_pred_pertubed = f(y_pred, noisy = True)
        local_loss.append(mean_squared_error(y_pred, y_local))
        global_loss.append(mean_squared_error(z_pred, z_global))
        global_pertube_loss.append(mean_squared_error(z_pred_pertubed, z_global))

    # x and y are bounds, so z should be the value *inside* those bounds.
    # Therefore, remove the last value from the z array.
    local_loss = np.array(local_loss).reshape(theta_0_range.shape)
    z_min, z_max = (local_loss).min(), np.abs(local_loss).max()

    fig, ax = plt.subplots(1,2)

    c = ax[0].pcolormesh(theta_0_range, theta_1_range, local_loss, cmap='RdBu', vmin=z_min, vmax=z_max)
    ax[0].set_title('local loss')
    # set the limits of the plot to the limits of the data
    ax[0].axis([theta_0_range.min(), theta_0_range.max(), theta_1_range.min(), theta_1_range.max()])
    fig.colorbar(c, ax=ax[0])

    # global_loss = np.array(global_loss).reshape(theta_0_range.shape)
    # c = ax[1].pcolormesh(theta_0_range, theta_1_range, global_loss, cmap='RdBu', vmin=z_min, vmax=z_max)
    # ax[1].set_title('system loss')
    # # set the limits of the plot to the limits of the data
    # ax[1].axis([theta_0_range.min(), theta_0_range.max(), theta_1_range.min(), theta_1_range.max()])
    # fig.colorbar(c, ax=ax[1])

    global_pertube_loss = np.array(global_pertube_loss).reshape(theta_0_range.shape)
    c = ax[1].pcolormesh(theta_0_range, theta_1_range, global_pertube_loss, cmap='RdBu', vmin=z_min, vmax=z_max)
    ax[1].set_title('system loss with pertubed component')
    # set the limits of the plot to the limits of the data
    ax[1].axis([theta_0_range.min(), theta_0_range.max(), theta_1_range.min(), theta_1_range.max()])
    fig.colorbar(c, ax=ax[1])

    return plt, fig, ax