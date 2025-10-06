df_alt = df_true.copy()

print(f'before: {df_true.shape[0]} observations, {df_true.shape[1]} variables')
df_true = df_true.drop('ratio',axis=1)
I = df_true['income'] > 1.5
df_true = df_true.drop(df_true[I].index)
df_true = df_true.drop(df_true.loc[:5].index)
print(f'after: {df_true.shape[0]} observations, {df_true.shape[1]} variables')

# alternative: keep where I is false
del df_alt['ratio']
I = df_alt['income'] > 1.5
df_alt = df_alt[~I]
df_alt = df_alt.iloc[5:,:]
print(f'after (alt): {df_alt.shape[0]} observations, {df_alt.shape[1]} variables')