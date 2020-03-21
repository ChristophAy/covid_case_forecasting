import os
import re
import logging

from git_root import git_root
import pandas as pd

pd.set_option("display.max_columns", 100)

# clone it from https://github.com/pcm-dpc/COVID-19"
data_folder = "C:\\Users\\matth\\repos\\COVID-19\\dati-regioni"

output_file = os.path.join(git_root(), "data", "df_it.csv")

dict_renaming = {
    "data": "date",
    "stato": "country",
    "codice_regione": "region_code",
    "denominazione_regione": "region_name",
    "lat": "region_latitude",
    "long": "region_longitude",
    "ricoverati_con_sintomi": "cases_pos_hospitalized_non_icu",
    "terapia_intensiva": "cases_pos_hospitalized_icu",
    "totale_ospedalizzati": "cases_pos_hospitalized",
    "isolamento_domiciliare": "cases_pos_in_home_isolation",
    "totale_attualmente_positivi": "cases_pos_total",
    "nuovi_attualmente_positivi": "cases_pos_new",
    "dimessi_guariti": "cases_recovered",
    "deceduti": "cases_deceased",
    "totale_casi": "cases_total",
    "tamponi": "tests_total",
}
dict_agg = {
    "cases_pos_hospitalized_non_icu": "sum",
    "cases_pos_hospitalized_icu": "sum",
    "cases_pos_hospitalized": "sum",
    "cases_pos_in_home_isolation": "sum",
    "cases_pos_total": "sum",
    "cases_pos_new": "sum",
    "cases_recovered": "sum",
    "cases_deceased": "sum",
    "cases_total": "sum",
    "tests_total": "sum",
}

all_files = [file for file in os.listdir(data_folder)]
exclude_files = ["dpc-covid19-ita-regioni-latest.csv", "dpc-covid19-ita-regioni.csv"]
data_files = list(set(all_files) - set(exclude_files))

# get time stamp
df_file_date = (
    pd.DataFrame(
        {
            "file": data_files,
            "date": [pd.to_datetime(re.findall(r"\d+", s)[1]) for s in data_files],
        }
    )
    .sort_values("date")
    .reset_index(drop=True)
)
df = pd.DataFrame()

for row in df_file_date.iterrows():
    logging.info(f"++ Reading in data for {row[1]['date']}")
    df_date = pd.read_csv(os.path.join(data_folder, row[1]["file"]))
    df_date = df_date.rename(columns=dict_renaming)
    df = pd.concat([df, df_date])

df = df.sort_values("date").reset_index(drop=True)
logging.info("+ Preparing data.")

for col in df.columns[df.columns.str.startswith("cases")]:
    col_neg = df[col] < 0
    if col_neg.any():
        logging.info(f"--- {col}: setting {col_neg.sum()} negative cases to zero.")
        df.loc[col_neg, col] = 0

logging.info("+ Correcting the total case data.")


def _get_total_cases(data):
    cases_daily = (
        data["cases_deceased"].diff()
        + data["cases_recovered"].diff()
        + data["cases_pos_new"]
    )
    cases_daily[0] = (
        data["cases_deceased"][0]
        + data["cases_recovered"][0]
        + data["cases_pos_new"][0]
    )
    return cases_daily.cumsum()


df_total = (
    df.groupby("region_name")
    .apply(_get_total_cases)
    .reset_index()
    .drop(columns="region_name")
    .rename(columns={"level_1": "index", 0: "cases_total"})
)
df = (
    df.drop(columns="cases_total")
    .merge(df_total, how="left", left_index=True, right_on="index")
    .drop(columns="index")
)

logging.info(f"+ Persisting file under {output_file}")
df.to_csv(output_file)
