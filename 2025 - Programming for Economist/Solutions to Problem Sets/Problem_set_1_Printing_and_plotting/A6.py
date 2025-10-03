import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import A1, A5

def plot(x1_vec,x2_vec,alpha,beta,uvals):

    # a. calculate
    x1_grid,x2_grid = np.meshgrid(x1_vec,x2_vec,indexing='ij')
    u_grid = A1.u_func(x1_grid,x2_grid,alpha=alpha,beta=beta)

    # b. plot
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1,projection='3d')

    # i. main
    h = ax.plot_surface(x1_grid,x2_grid,u_grid,cmap=cm.jet)
    fig.colorbar(h)

    # ii. indifference curves
    for uval in uvals:

        I = np.log(x1_vec) > np.log(uval) + np.log(alpha)/beta
        x1_vec_ = x1_vec[I]
        x2_vec_ = A5.x2_func(x1_vec_,uval,alpha=alpha,beta=beta)

        J = x2_vec_ < x2_vec[-1] 
        uval_vec_ = uval*np.ones_like(x2_vec_[J])
        ax.plot(x1_vec_[J],x2_vec_[J],uval_vec_,lw=2,color='black',zorder=99)

    # c. add labels
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

    # c. invert xaxis to bring origin in center front
    ax.invert_xaxis()
    fig.tight_layout(pad=0.1)