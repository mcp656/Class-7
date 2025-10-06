I = (df_true['income'] < 0.5)
df_true.loc[I, ['consumption']] = df_true.loc[I, ['income']].values
df_true['consumption'].mean()