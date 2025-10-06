I = (df_true['income'] < 0.5)
df_true.loc[I, ['consumption']] = 0.5
df_true['consumption'].mean()