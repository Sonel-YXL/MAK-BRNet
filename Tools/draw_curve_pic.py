import matplotlib.pyplot as plt
import numpy as np
import os

def load_data(filename):
    pts = []
    f = open(filename, "rb")
    for line in f:
        pts.append(float(line.strip()))
    f.close()
    return pts




train_pts = load_data("path")

def draw_loss():
    x = np.linspace(0, len(train_pts), len(train_pts))
    plt.plot(x,train_pts,'ro-',label='train')


    plt.legend(loc='upper right')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.show()


def draw_loss_ap():
    ap05_pts = load_data(os.path.join(weights_path, 'ap_list.txt'))

    x = np.linspace(0,len(train_pts),len(train_pts))
    x1 = np.linspace(0, len(train_pts), len(ap05_pts))

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(x, train_pts, 'ro-',label='train')
    ax1.tick_params(axis='y', labelcolor=color)
    plt.legend(loc = 'lower right')
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('AP', color=color)
    ax2.plot(x1, ap05_pts, 'go-',label='AP@05')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.legend(loc = 'upper right')
    plt.show()

def draw_curve(y_name_list, y_list):
    ap05_pts = load_data(os.path.join(weights_path, 'ap_list.txt'))

    x = np.linspace(0,len(train_pts),len(train_pts))
    x1 = np.linspace(0, len(train_pts), len(ap05_pts))

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(x, train_pts, 'ro-',label='train')
    ax1.tick_params(axis='y', labelcolor=color)
    plt.legend(loc = 'lower right')
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('AP', color=color)
    ax2.plot(x1, ap05_pts, 'go-',label='AP@05')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.legend(loc = 'upper right')
    plt.show()


if __name__ == '__main__':

    draw_loss_ap()