model = ExchangeEconomyQuasiLineaModelClass()
par = model.par
sol = model.sol
sim = model.sim

fig,ax_A,ax_B = model.create_edgeworthbox()

ax_A.scatter(par.w1A,par.w2A,marker='s',color='black',label='endowment',zorder=3)

model.indifference_curve_A(ax_A,par.w1A,par.w2A,color=colors[0],label='A')
model.indifference_curve_A(ax_A,par.w1A*0.9,par.w2A,color=colors[0],label=None,ls='--')
model.indifference_curve_A(ax_A,par.w1A*1.1,par.w2A,color=colors[0],label=None,ls='--')

model.indifference_curve_B(ax_B,1-par.w1A,1-par.w2A,color=colors[1],label='B')
model.indifference_curve_B(ax_B,1-par.w1A*0.9,1-par.w2A*0.9,color=colors[1],label=None,ls='--')
model.indifference_curve_B(ax_B,1-par.w1A*1.1,1-par.w2A*1.1,color=colors[1],label=None,ls='--')

model.plot_improvement_set(ax_A)

model.add_legend(ax_A,ax_B)