ages = [f'{age} years' for age in range(18,65+1)]

working_pop = BEFOLK1[BEFOLK1.ALDER.isin(ages)].groupby('year').sum()
working_pop = working_pop.drop(columns=['ALDER'])
working_pop = working_pop.rename(columns={'population':'working_population'})

merged_true = pd.merge(nah1_true, working_pop, how='left', on=['year'])
merged_true = pd.merge(merged_true, pop, how='left', on=['year'])

fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(1,1,1)

merged_true['value_pop'] = merged_true.value/merged_true.population
merged_true['value_working_pop'] = merged_true.value/merged_true.working_population

I = merged_true.unit == 'real' 
I &= merged_true.variable == 'Y' 

df = merged_true[I].set_index('year')
ax.plot(df.value_pop, label='per capita')
ax.plot(df.value_working_pop, label='per working-age')

ax.legend()