import torch as th
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.models import EpidemicModel, EpidemicModelUnitRoot, PoissonLogLikelihood


def train_model(
    x,
    y,
    x_t,
    seed=2,
    max_grad=10,
    learning_rate=1e-3,
    n_epochs=3000,
    print_every=100,
    which_model="stationary",
):
    """Train the epidemic model with PyTorch on aggregate data."""

    th.manual_seed(seed)
    np.random.seed(seed)

    if which_model == "stationary":
        model = EpidemicModel()
    elif which_model == "unit_root":
        model = EpidemicModelUnitRoot()
    else:
        raise ValueError("'which_model' must be 'stationary' or 'unit_root'.")
    criterion = PoissonLogLikelihood()
    optimizer = th.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(n_epochs):

        X_batch = th.tensor(x).float()
        y_batch = th.tensor(y).float()
        new_cases = th.tensor(x_t).float()

        optimizer.zero_grad()

        log_lam, _ = model(X_batch, new_cases)
        # import pdb; pdb.set_trace()
        loss = criterion(log_lam, y_batch)
        loss.backward()
        _ = th.nn.utils.clip_grad_norm_(model.parameters(), max_grad)

        optimizer.step()

        if epoch % print_every == 0:
            train_error = loss.detach().numpy()
            print("epoch = %d, train error = %.5f" % (epoch, train_error))

    return model


def train_model_region(
    df,
    scale_by=100,
    seed=2,
    max_grad=10,
    learning_rate=1e-3,
    n_epochs=3000,
    print_every=100,
    which_model="stationary",
):
    """Train the epidemic model with PyTorch on regional data."""

    th.manual_seed(seed)
    np.random.seed(seed)

    x_region, x_t_region, y_region = prepare_region_data(df, scale_by=scale_by)

    if which_model == "stationary":
        model = EpidemicModel()
    elif which_model == "unit_root":
        model = EpidemicModelUnitRoot()
    else:
        raise ValueError("'which_model' must be 'stationary' or 'unit_root'.")
    criterion = PoissonLogLikelihood()
    optimizer = th.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(n_epochs):

        for region in x_region.keys():
            x = x_region[region]
            y = y_region[region]
            x_t = x_t_region[region]

            X_batch = th.tensor(x).float()
            y_batch = th.tensor(y).float()
            new_cases = th.tensor(x_t).float()

            optimizer.zero_grad()

            log_lam, _ = model(X_batch, new_cases)
            loss = criterion(log_lam, y_batch)
            loss.backward()
            _ = th.nn.utils.clip_grad_norm_(model.parameters(), max_grad)

            optimizer.step()

        if epoch % print_every == 0:
            train_error = loss.detach().numpy()
            print("epoch = %d, train error = %.5f" % (epoch, train_error))

    return model


def prepare_region_data(df, scale_by=1):
    """Prepare regional data so that it can be used in train_model_region."""
    regions = df.region_name.unique()
    x_region = {}
    x_t_region = {}
    y_region = {}

    for i, region in enumerate(regions):
        x_region[region] = (
            df.loc[
                (df.region_name == region) & (df["cases_pos_hospitalized_icu"] > 0),
                "cases_pos_hospitalized_icu",
            ].to_numpy()[:-1]
            / scale_by
        )
        x_t_region[region] = (
            df.loc[
                (df.region_name == region) & (df["cases_pos_hospitalized_icu"] > 0),
                "cases_pos_hospitalized_icu_change",
            ].to_numpy()[1:]
            / scale_by
        )
        y_region[region] = (
            df.loc[
                (df.region_name == region) & (df["cases_pos_hospitalized_icu"] > 0),
                "cases_pos_hospitalized_icu_change",
            ].to_numpy()[1:]
            / scale_by
        )

    return x_region, x_t_region, y_region


def predict_vs_actual_by_region(df, model_region, scale_by=1):
    """Compute predicted vs. actual plots by region."""

    x_region, x_t_region, y_region = prepare_region_data(df, scale_by=scale_by)

    dfs = []

    for region in x_region.keys():
        x = x_region[region]
        y = y_region[region]
        x_t = x_t_region[region]
        lam, omega_pred = model_region(th.tensor(x).float(), th.tensor(x_t).float())
        df_region = df.loc[
            (df.region_name == region) & (df["cases_pos_hospitalized_icu"] > 0)
        ]
        df_region = df_region.iloc[1:]
        df_region["cases_icu_act"] = scale_by * y
        df_region["cases_icu_predicted"] = scale_by * np.exp(lam.detach().numpy())
        dfs.append(df_region)

    df_pred_act = pd.concat(dfs)

    return df_pred_act


def plot_pred_vs_act_multistep(df, model_region, region_pred, horizon, scale_by):
    """Plot predicted vs. actual plots for multistep predictions, for one region."""
    df_pred = df[df.region_name == region_pred]
    x = df_pred["cases_pos_hospitalized_icu"].to_numpy()[:-1] / scale_by
    x_t = df_pred["cases_pos_hospitalized_icu_change"].to_numpy()[:-1] / scale_by
    y = df_pred["cases_pos_hospitalized_icu_change"].to_numpy()[1:] / scale_by

    n = len(x)
    start_pred = n - horizon

    pred = model_region.predict(
        th.tensor(x[: (n - horizon)]).float(),
        th.tensor(x_t[: (n - horizon)]).float(),
        horizon=horizon,
    )

    df_pred = pd.DataFrame(
        {
            "time since first icu case": np.arange(1, 1 + len(pred)),
            "predicted increase in icu cases": pred * scale_by,
            "actual increase in icu cases": y * scale_by,
        }
    )
    df_pred = pd.melt(
        df_pred,
        id_vars="time since first icu case",
        var_name="icu cases",
        value_name="number of cases",
    )

    df_interval = pd.DataFrame(
        {
            "time since first icu case": np.arange(1, 1 + len(pred)),
            "lower": (pred * scale_by - 2 * np.sqrt(pred * scale_by)),
            "upper": (pred * scale_by + 2 * np.sqrt(pred * scale_by)),
        }
    )

    col_pred = sns.color_palette("Set1", n_colors=2)[1]
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.lineplot(
        ax=ax,
        data=df_pred,
        x="time since first icu case",
        y="number of cases",
        hue="icu cases",
        palette=sns.color_palette("Set1", n_colors=2)[::-1],
    )
    ax.vlines(
        x=start_pred,
        ymin=df_interval["lower"][start_pred - 1],
        ymax=df_interval["upper"][start_pred - 1],
        colors="red",
        color=col_pred,
        linestyle="--",
    )
    ax.plot(
        df_interval["time since first icu case"][start_pred - 1 :],
        df_interval["lower"][start_pred - 1 :],
        color=col_pred,
        linestyle="--",
    )
    ax.plot(
        df_interval["time since first icu case"][start_pred - 1 :],
        df_interval["upper"][start_pred - 1 :],
        color=col_pred,
        linestyle="--",
    )
    return fig, ax


def plot_pred_vs_act_one_step(df_pred_act, region_pred):
    """Plot predicted vs. actual plots for one-step-ahead predictions, for one region."""
    df_pred_wide = (
        df_pred_act[["cases_icu_act", "cases_icu_predicted"]]
        .loc[df_pred_act["region_name"] == region_pred]
        .rename(
            columns={
                "cases_icu_act": "actual increase in icu cases",
                "cases_icu_predicted": "predicted increase in icu cases",
            }
        )
    )
    df_pred_wide["t"] = np.arange(0, df_pred_wide.shape[0])
    df_pred = pd.melt(df_pred_wide, id_vars="t")

    pred = df_pred_wide["predicted increase in icu cases"].to_numpy()
    df_interval = pd.DataFrame(
        {
            "time since first icu case": np.arange(0, len(pred)),
            "lower": (pred - 2 * np.sqrt(pred)),
            "upper": (pred + 2 * np.sqrt(pred)),
        }
    )

    col_pred = sns.color_palette("Set1", n_colors=2)[1]
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.lineplot(
        ax=ax,
        data=df_pred,
        x="t",
        y="value",
        hue="variable",
        palette=sns.color_palette("Set1", n_colors=2),
    )
    ax.set_ylabel("number of cases")
    ax.set_xlabel("time since first icu case")
    ax.plot(
        df_interval["time since first icu case"],
        df_interval["lower"],
        color=col_pred,
        linestyle="--",
    )
    ax.plot(
        df_interval["time since first icu case"],
        df_interval["upper"],
        color=col_pred,
        linestyle="--",
    )
    return fig, ax
