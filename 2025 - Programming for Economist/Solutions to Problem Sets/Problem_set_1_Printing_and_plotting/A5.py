import numpy as np
import matplotlib.pyplot as plt # baseline module
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def x2_func(x1,u,alpha=0.5,beta=0.5):
    return ((u**(-beta)-alpha*x1**(-beta))/(1-alpha))**(-1/beta)

def plot(alpha,betas,uvals):

    x1_vec = np.linspace(0.10,4.0,1000)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    # main
    ls = ['-','--',':']
    for i,beta in enumerate(betas):
        for j,uval in enumerate(uvals):
            
            I = np.log(x1_vec) > np.log(uval) + np.log(alpha)/beta
            x2_vec = x2_func(x1_vec[I],uval,beta=beta)

            if i == 0:
                label = fr'$u = {uval}$, $\beta = {beta}$'
            elif j == 0:
                label = fr'$\beta = {beta}$'
            else:
                label = ''

            ax.plot(x1_vec[I],x2_vec,ls=ls[i],color=colors[j],label=label)

    # add 45 degree line
    ax.plot([0,4],[0,4],color='grey',zorder=0)

    # labels
    ax.set_title('Indifference Curves',pad=10)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

    # lims
    ax.set_ylim(-0.1,4)
    ax.set_xlim(-0.1,4)

    # legend
    ax.legend(ncol=2,loc='lower left',prop={'size': 10})

    # save
    fig.savefig('A5_indifference_curves.png')