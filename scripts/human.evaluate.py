import glob
import json

import pandas
import polars
import rich
from statsmodels.stats.inter_rater import fleiss_kappa

SAMPLES: list[str] = ["lux", "deu"]


for sample in SAMPLES:
    print(f"Evaluating Language: {sample}")

    raw = [
        json.loads(open(file).read())
        for file in glob.glob(f"data/human/{sample}/*.json")
    ]

    demographics = pandas.concat(
        [pandas.DataFrame(row["survey"]).set_index("id").T for row in raw]
    ).reset_index(drop=True)
    rich.print(demographics.head())

    rich.print(demographics["employment_status"].value_counts())

    annotations = (
        pandas.concat(
            [
                pandas.concat(
                    [
                        (
                            pandas.json_normalize(annotation["annotations"])
                            .assign(sample=annotation["id"])
                            .set_index("sample")
                        )
                        for annotation in row["annotations"]
                    ]
                )
                .assign(participant=n)
                .set_index("participant", append=True)
                for n, row in enumerate(raw)
            ]
        )
        .reset_index()
        .drop(columns=["id"])
        .pipe(lambda _df: _df.assign(sample=_df["sample"].str.zfill(2)))
    )

    annotations_wide = (
        annotations.groupby("sample")["response"]
        .value_counts()
        .reset_index()
        .pivot(index="sample", columns="response", values="count")[
            ["Yes", "No", "None"]
        ]
        .fillna(0)
        .sort_index()
    )

    polars.from_pandas(annotations_wide).describe().write_csv(
        f"reports/evaluate.human.{sample}.csv"
    )
    rich.print(f"{fleiss_kappa(annotations_wide.to_numpy(), method="randolph")=}")
