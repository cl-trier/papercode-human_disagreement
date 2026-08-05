import glob

import polars

results = []


for file in glob.glob("reports/evaluate.*.*.csv"):
    meta = file.split("/")[-1].split(".")
    results.append(
        polars.read_csv(file)
        .select(
            polars.lit(meta[1]).alias("subset"),
            polars.lit(meta[2]).alias("lang"),
            polars.all(),
        )
        .filter(~polars.col("statistic").is_in(["count", "null_count"]))
    )


polars.concat(results).write_csv("reports/evaluate.all.csv")
