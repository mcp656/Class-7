model.solve_dictator_B()

par = model.par
sol = model.sol

fig,ax_A,ax_B = model.create_edgeworthbox()

ax_A.scatter(par.w1A,par.w2A,marker='s',color='black',label='endowment')
ax_A.scatter(sol.xA[0],sol.xA[1],marker='o',color='black',label='equilibrium')
ax_A.scatter(sol.xA_dictatorA[0],sol.xA_dictatorA[1],marker='x',color='black',label='dictatorA')
ax_A.scatter(sol.xA_dictatorB[0],sol.xA_dictatorB[1],marker='x',color='black',label='dictatorB')

model.indifference_curve_A(ax_A,sol.xA_dictatorA[0],sol.xA_dictatorA[1],color=colors[0])
model.indifference_curve_B(ax_B,1-sol.xA_dictatorA[0],1-sol.xA_dictatorA[1],color=colors[1])

model.indifference_curve_A(ax_A,sol.xA_dictatorB[0],sol.xA_dictatorB[1],color=colors[0],ls='--')
model.indifference_curve_B(ax_B,1-sol.xA_dictatorB[0],1-sol.xA_dictatorB[1],color=colors[1],ls='--')

model.plot_improvement_set(ax_A)
model.add_legend(ax_A,ax_B)
