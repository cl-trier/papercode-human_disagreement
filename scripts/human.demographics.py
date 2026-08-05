import glob
import json

import pandas
import rich

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

    rich.print(demographics["gender"].value_counts())
    rich.print(demographics["religion"].value_counts())
