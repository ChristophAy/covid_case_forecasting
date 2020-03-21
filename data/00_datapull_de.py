import pandas as pd
import os

# Wikipedia manual pull

rename_dict_demographic = {
    "Land": "region_code",
    "lat": "region_latitude",
    "long": "region_longitude",
    "Name": "region_name",
}

path = os.path.join(os.getcwd(), "data")
infected = pd.read_csv(os.path.join(path, "confirmed_infections_RKI_DE.csv"))
infected["Date"] = pd.to_datetime(infected["Date"], dayfirst=True)
infected = pd.melt(
    infected, id_vars="Date", var_name="region_code", value_name="cases_pos_total"
)
infected.rename(columns={"Date": "date"}, inplace=True)
deaths = pd.read_csv(os.path.join(path, "deaths_RKI_DE.csv"))
deaths["Date"] = pd.to_datetime(deaths["Date"], dayfirst=True)
deaths = pd.melt(
    deaths, id_vars="Date", var_name="region_code", value_name="cases_deceased"
)
deaths.rename(columns={"Date": "date"}, inplace=True)
demographic_info = pd.read_csv(os.path.join(path, "demographic_info.csv"))
demographic_info.rename(columns=rename_dict_demographic, inplace=True)

max_inf = infected["date"].max()
max_dea = deaths["date"].max()
min_max_date = min(max_inf, max_dea)
combined = pd.merge(
    infected.loc[infected["date"] <= min_max_date],
    deaths.loc[deaths["date"] <= min_max_date],
    left_on=["date", "region_code"],
    right_on=["date", "region_code"],
    how="left",
)
combined.fillna(0, inplace=True)
combined = pd.merge(
    combined,
    demographic_info,
    left_on="region_code",
    right_on="region_code",
    how="left",
)
combined["country"] = "DE"

combined["cases_pos_new"] = 0
for c in combined["region_code"].unique():
    combined["cases_pos_new"].loc[combined["region_code"] == c] = (
        combined["cases_pos_total"].loc[combined["region_code"] == c].diff()
    )

combined.to_csv(os.path.join(path, "DE_combined_wiki.csv"), index=False)

# Pull from
# https://npgeo-corona-npgeo-de.hub.arcgis.com/datasets/dd4580c810204019a7b8eb3e0b329dd6_0/data?orderBy=AnzahlFall&orderByAsc=false

LOAD_FROM_URL = True

if LOAD_FROM_URL:
    url = "https://opendata.arcgis.com/datasets/dd4580c810204019a7b8eb3e0b329dd6_0.geojson"
    out = pd.read_json(url)

    keys = out["features"].iloc[0]["properties"].keys()
    new_df_dict = {}
    for k in keys:
        new_df_dict[k] = []
    for i in range(out.shape[0]):
        for k in keys:
            new_df_dict[k].append(out["features"].iloc[i]["properties"][k])
    case_data = pd.DataFrame(new_df_dict)
else:
    path = os.path.join(os.getcwd(), "data")
    case_data = pd.read_csv(os.path.join(path, "RKI_COVID19.csv"))

case_data["date"] = pd.to_datetime(case_data["Meldedatum"])

land_aggregated = (
    case_data[["date", "IdBundesland", "AnzahlFall", "AnzahlTodesfall"]]
    .groupby(["date", "IdBundesland"])
    .sum()
    .reset_index()
)
land_aggregated = land_aggregated.loc[land_aggregated["IdBundesland"] > 0]
land_aggregated.sort_values(["IdBundesland", "date"], inplace=True)

land_aggregated["cases_pos_total"] = 0
land_aggregated["cases_deceased"] = 0
for c in land_aggregated["IdBundesland"].unique():
    land_aggregated["cases_pos_total"].loc[land_aggregated["IdBundesland"] == c] = (
        land_aggregated["AnzahlFall"].loc[land_aggregated["IdBundesland"] == c].cumsum()
    )
    land_aggregated["cases_deceased"].loc[land_aggregated["IdBundesland"] == c] = (
        land_aggregated["AnzahlTodesfall"]
        .loc[land_aggregated["IdBundesland"] == c]
        .cumsum()
    )


rename_dict = {"AnzahlFall": "cases_pos_new", "AnzahlTodesfall": "cases_deceased_new"}

land_aggregated.rename(columns=rename_dict, inplace=True)
land_aggregated = pd.merge(
    land_aggregated, demographic_info, left_on="IdBundesland", right_on="Id", how="left"
)
land_aggregated.to_csv(os.path.join(path, "DE_combined_bottom_up.csv"), index=False)

