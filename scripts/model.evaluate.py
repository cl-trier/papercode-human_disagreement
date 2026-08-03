import polars
from statsmodels.stats.inter_rater import fleiss_kappa

SAMPLES: list[str] = ["lux", "deu"]


for sample in SAMPLES:
    print(f"Evaluating Language: {sample}")

    data = polars.read_csv(f"data/model/base.{sample}.csv")

    data_agg = data.select(
        *[
            polars.sum_horizontal(polars.col("*").exclude("idx") == label)
            .cast(float)
            .alias(label)
            for label in ["Yes", "No", "None"]
        ]
    )

    print(data_agg.describe())
    print(f"{fleiss_kappa(data_agg.to_numpy(), method="randolph")=}")
