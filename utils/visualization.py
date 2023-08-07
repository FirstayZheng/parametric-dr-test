import matplotlib.pyplot as plt
import os

def plot_2D(points, color=None,cmap=None, savePath='./outputs', title=None):
    x, y = points.T
    fig, ax = plt.subplots()
    if color is not None:
        # ax.scatter(x, y, s=1, c=color, cmap=cmap)
        ax.scatter(x, y, s=3, c=color, cmap=cmap)
    else:
        # ax.scatter(x, y, s=1)
        ax.scatter(x, y, s=3)
    ax.set_title(title)
    if not os.path.exists(f'{savePath}'):
        os.makedirs(f'{savePath}')
    fig.savefig(f'{savePath}/{title}.png')

def plot_original_3D(points, color=None, cmap=None, savePath='./outputs', title=None):
    x, y, z = points.T
    fig, ax = plt.subplots(
        subplot_kw={'projection': '3d'},
    )

    # ax.scatter(x, y, z, s=1.0, c=color, cmap=cmap)
    ax.scatter(x, y, z, s=3.0, c=color, cmap=cmap)
    ax.set_title(f'{title}_original')
    if not os.path.exists(f'{savePath}'):
        os.makedirs(f'{savePath}')
    fig.savefig(f'{savePath}/{title}_original.png')


def plot_loss_curve(
    loss_value,
    savePath='./outputs',
):
    fig, ax = plt.subplots()
    epoch = list(range(len(loss_value)))
    ax.plot(epoch, loss_value)
    ax.set(xlabel='epoch', ylabel='loss', title='loss curve')
    ax.grid()

    fig.savefig(f'{savePath}/loss_curve.png')


def plot_small_multiples(points_list,
                  ncols,
                  nrows,
                  color=None,
                  savePath='./outputs',
                  title=None):
    fig, ax = plt.subplots(
        ncols=ncols,
        nrows=nrows,
        figsize=(5 * ncols, 5 * nrows),
    )

    for p, (batch, point) in enumerate(points_list):
        if p >= ncols * nrows:
            break
        _x, _y = p // ncols, p % ncols
        x, y = point.T
        ax[_x, _y].scatter(
            x,
            y,
            c=color,
            # s=2,
            s=5,
        )
        ax[_x, _y].set_title(str(batch))
    fig.suptitle(title)

    if not os.path.exists(f'{savePath}'):
        os.makedirs(f'{savePath}')
    fig.savefig(f'{savePath}/{title}.png')
