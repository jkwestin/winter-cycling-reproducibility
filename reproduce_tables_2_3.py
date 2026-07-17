#!/usr/bin/env python3

# Copyright (c) 2026 Jonas Westin and Per Åhag
# SPDX-License-Identifier: MIT
#
# This file is part of the replication materials for:
# "Who Cycles in Winter? Seasonal Cycling and Mode Share Change
# in Northern Sweden."
#
# Licensed under the MIT License. See LICENSE in the repository root.

from pathlib import Path
from datetime import datetime
from time import perf_counter
import argparse

import numpy as np
import pandas as pd
from scipy.stats import norm

import biogeme.biogeme as bio
import biogeme.database as db
import biogeme.models as models
from biogeme.expressions import Beta, Variable


MODES = {1: "CAR", 2: "BIKE", 3: "WALK", 4: "BUS"}
NONCAR = {2: "BIKE", 3: "WALK", 4: "BUS"}
SMALL = ["winter", "female", "female_winter", "distance_km"]
FULL = [
    "winter", "distance_km", "female", "female_winter",
    "purpose_work_education", "purpose_shopping_errands",
    "purpose_leisure_social", "weekday", "morning_peak",
    "afternoon_peak", "car_access", "bike_access",
    "age_16_24", "age_45_64", "age_65plus", "student",
    "edu_higher_medium", "edu_primary", "edu_other",
]

START = perf_counter()


def progress(text):
    print(
        f"[{datetime.now():%H:%M:%S}] {text} "
        f"({perf_counter() - START:.1f} s)",
        flush=True,
    )


def prepare(path):
    progress(f"Reading {path}")
    d = pd.read_csv(path, sep=";", low_memory=False)
    d.columns = d.columns.astype(str).str.strip()

    numeric = [
        "wave", "distance_km", "female", "hour", "weekday",
        "car_access", "bike_access", "birth_year",
        "student", "weight_population",
    ]
    d[numeric] = d[numeric].apply(pd.to_numeric, errors="coerce")

    for col, case in [
        ("mode4", "upper"),
        ("trip_purpose", "lower"),
        ("education", "upper"),
    ]:
        d[col] = getattr(
            d[col].astype("string").str.strip().str,
            case,
        )()

    d = d[
        d["wave"].isin([2022, 2025])
        & d["mode4"].isin(MODES.values())
        & d["distance_km"].between(0, 50, inclusive="right")
        & (d["weight_population"] > 0)
    ].copy()

    d["CHOICE"] = d["mode4"].map(
        {"CAR": 1, "BIKE": 2, "WALK": 3, "BUS": 4}
    )
    d["winter"] = (d["wave"] == 2025).astype(int)
    d["female_winter"] = d["female"] * d["winter"]

    binaries = {
        "purpose_work_education": d["trip_purpose"] == "work_education",
        "purpose_shopping_errands": d["trip_purpose"] == "shopping_errands",
        "purpose_leisure_social": d["trip_purpose"] == "leisure_social",
        "morning_peak": d["hour"].between(7, 9),
        "afternoon_peak": d["hour"].between(15, 17),
        "student": d["student"] == 1,
        "edu_higher_medium": d["education"] == "HIGHER_MEDIUM",
        "edu_primary": d["education"] == "PRIMARY",
        "edu_other": d["education"] == "OTHER",
    }

    age = d["wave"] - d["birth_year"]
    binaries.update({
        "age_16_24": age.between(16, 24),
        "age_45_64": age.between(45, 64),
        "age_65plus": age >= 65,
    })

    for name, values in binaries.items():
        d[name] = values.fillna(False).astype(int)

    progress(
        f"Prepared {len(d):,} trips from "
        f"{d['person_id'].astype(str).nunique():,} persons"
    )
    return d


def make_table2(d):
    progress("Calculating Table 2")
    d = d.copy()
    d["Overall"] = "Overall"
    d["Gender"] = d["female"].map({0: "Men", 1: "Women"})
    d["Purpose"] = d["trip_purpose"].map({
        "work_education": "Work/education",
        "shopping_errands": "Shopping/errands",
        "leisure_social": "Leisure/social",
    })

    out = []
    for group in ["Overall", "Gender", "Purpose"]:
        x = (
            d.dropna(subset=[group])
            .groupby([group, "wave", "mode4"])["weight_population"]
            .sum()
            .rename("n")
            .reset_index()
        )
        x["share"] = 100 * x["n"] / x.groupby(
            [group, "wave"]
        )["n"].transform("sum")
        x = x.pivot(
            index=[group, "mode4"],
            columns="wave",
            values="share",
        )
        x["change_pp"] = x[2025] - x[2022]
        out.append(
            x["change_pp"]
            .unstack("mode4")
            .reset_index()
            .rename(columns={group: "Group"})
        )

    progress("Finished Table 2")
    return pd.concat(out).reindex(
        columns=["Group", "CAR", "BIKE", "WALK", "BUS"]
    ).round(2)


def sample_for_model(d, variables):
    cols = ["CHOICE", "person_id", "weight_population", *variables]
    s = d[cols].copy()
    s[["CHOICE", "weight_population", *variables]] = s[
        ["CHOICE", "weight_population", *variables]
    ].apply(pd.to_numeric, errors="coerce")
    s = s.dropna().reset_index(drop=True)
    s["person_id"] = s["person_id"].astype(str)
    s["CHOICE"] = s["CHOICE"].astype(int)
    n = s.groupby("person_id")["person_id"].transform("size")
    s["weight"] = s["weight_population"] / n
    s["weight"] /= s["weight"].mean()
    progress(
        f"Model sample: {len(s):,} trips, "
        f"{s['person_id'].nunique():,} persons"
    )
    return s


def estimate(s, variables, name, threads):
    progress(f"Estimating {name}")
    database = db.Database(
        name,
        s[["CHOICE", "weight", *variables]],
    )
    x = {v: Variable(v) for v in variables}
    utility = {1: Beta("ASC_CAR", 0, None, None, 1)}

    for alt, mode in NONCAR.items():
        utility[alt] = Beta(
            f"ASC_{mode}", 0, None, None, 0
        ) + sum(
            Beta(f"B_{v}_{mode}", 0, None, None, 0) * x[v]
            for v in variables
        )

    biogeme = bio.BIOGEME(
        database,
        Variable("weight") * models.loglogit(
            utility,
            {k: 1 for k in MODES},
            Variable("CHOICE"),
        ),
        number_of_threads=threads,
    )
    biogeme.model_name = str(name)
    biogeme.modelName = str(name)
    biogeme.generate_html = False
    biogeme.generate_yaml = False

    result = biogeme.estimate()
    progress(f"Finished {name}")
    return {
        str(k): float(v)
        for k, v in result.get_beta_values().items()
    }


def clustered_table(s, variables, estimates):
    progress("Calculating clustered standard errors")
    x = np.column_stack([
        np.ones(len(s)),
        s[variables].to_numpy(float),
    ])
    y = s["CHOICE"].to_numpy(int)
    w = s["weight"].to_numpy(float)
    k = x.shape[1]

    names = [
        name
        for mode in NONCAR.values()
        for name in [f"ASC_{mode}"] + [
            f"B_{v}_{mode}" for v in variables
        ]
    ]
    beta = np.array([
        [estimates[f"ASC_{mode}"]] + [
            estimates[f"B_{v}_{mode}"] for v in variables
        ]
        for mode in NONCAR.values()
    ])

    u = np.column_stack([np.zeros(len(s)), x @ beta.T])
    u -= u.max(axis=1, keepdims=True)
    p = np.exp(u)
    p /= p.sum(axis=1, keepdims=True)

    scores = np.column_stack([
        w[:, None] * ((y == alt) - p[:, alt - 1])[:, None] * x
        for alt in NONCAR
    ])

    bread = sum(
        w[i] * np.kron(
            np.diag(p[i, 1:]) - np.outer(p[i, 1:], p[i, 1:]),
            np.outer(x[i], x[i]),
        )
        for i in range(len(s))
    )

    meat_rows = (
        pd.DataFrame(scores)
        .assign(person_id=s["person_id"])
        .groupby("person_id")
        .sum()
        .to_numpy()
    )
    inv = np.linalg.pinv(bread)
    cov = inv @ (meat_rows.T @ meat_rows) @ inv

    values = beta.ravel()
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    z = np.divide(
        values,
        se,
        out=np.full_like(values, np.nan),
        where=se > 0,
    )

    progress("Finished clustered standard errors")
    return pd.DataFrame({
        "parameter": names,
        "coefficient": values,
        "clustered_se": se,
        "z": z,
        "p_value": 2 * norm.sf(np.abs(z)),
    })


def predictions(s, variables, results, model):
    progress(f"Calculating predictions for {model}")
    b = results.set_index("parameter")["coefficient"].to_dict()

    def probs(d):
        u = np.zeros((len(d), 4))
        for alt, mode in NONCAR.items():
            u[:, alt - 1] = b[f"ASC_{mode}"] + sum(
                b[f"B_{v}_{mode}"] * d[v].to_numpy(float)
                for v in variables
            )
        u -= u.max(axis=1, keepdims=True)
        p = np.exp(u)
        return p / p.sum(axis=1, keepdims=True)

    rows = []
    changes = {}

    for female, group in [(0, "Men"), (1, "Women")]:
        x = s[s["female"] == female].copy()
        means = []
        for winter in [0, 1]:
            z = x.copy()
            z["winter"] = winter
            z["female_winter"] = z["female"] * winter
            means.append(
                np.average(
                    probs(z),
                    axis=0,
                    weights=z["weight"],
                )
            )
        changes[group] = means[1] - means[0]

    changes["Women minus Men"] = changes["Women"] - changes["Men"]

    for group, change in changes.items():
        rows.extend({
            "model": model,
            "group": group,
            "mode": mode,
            "change_pp": 100 * change[alt - 1],
        } for alt, mode in MODES.items())

    progress(f"Finished predictions for {model}")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/synthetic_data_2022_2025.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output"),
    )
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    d = prepare(args.data)
    make_table2(d).to_csv(
        args.output / "table2.csv",
        index=False,
    )

    coefficients = []
    predicted = []

    for model, variables in [("small", SMALL), ("full", FULL)]:
        s = sample_for_model(d, variables)
        estimates = estimate(
            s,
            variables,
            f"table3_{model}",
            args.threads,
        )
        result = clustered_table(s, variables, estimates)
        result.insert(0, "model", model)
        coefficients.append(result)
        predicted.append(
            predictions(s, variables, result, model)
        )

    pd.concat(coefficients).to_csv(
        args.output / "table3_coefficients.csv",
        index=False,
    )
    pd.concat(predicted).to_csv(
        args.output / "table3_predictions.csv",
        index=False,
    )
    progress("All CSV files saved")


if __name__ == "__main__":
    main()