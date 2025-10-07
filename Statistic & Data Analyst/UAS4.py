from statsmodels.stats.contingency_tables import mcnemar

table = [[70, 15], [5, 25]]
result = mcnemar(table, exact=True)
print(f"Statistic: {result.statistic}, p-value: {result.pvalue}")