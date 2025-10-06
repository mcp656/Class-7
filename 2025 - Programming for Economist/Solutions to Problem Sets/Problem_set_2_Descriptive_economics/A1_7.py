deciles = df_true.quantile([0.1 * i for i in range(1, 10)])
df_true['decile'] = pd.cut(df_true['inc'], bins=[-np.inf] + list(deciles['inc']) + [np.inf], labels=range(1, 11))
income_share = df_true.groupby('decile')['inc'].sum() / df_true['inc'].sum()
display(income_share)