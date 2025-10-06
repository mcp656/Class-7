I = (df_true['income'] > 1) & (df_true['ratio'] > 0.7)
df_true.loc[I].head()