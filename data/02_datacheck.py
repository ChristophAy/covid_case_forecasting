import os
import logging

from git_root import git_root
import pandas as pd

input_file = os.path.join(git_root(), "data", "df_it.csv")

df = pd.read_csv(input_file)

logging.info("+ Check 1: cumulative case numbers should not decrease.")
df = df.sort_values("date").reset_index(drop=True)
df_check = (
    df.groupby("region_name")
    .apply(lambda dd: dd["cases_total"].diff())
    .reset_index()
    .dropna()
)

assert (
    df_check.cases_total >= 0
).all(), "Cumulative number of cases should not decrease over time."
