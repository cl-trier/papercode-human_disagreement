import json

import pandas
import vllm

import cultural_annotation as ca

if __name__ == "__main__":
    SAMPLE: str = "lux"

    SURVEY = json.load(open(f"data/survey.{SAMPLE}.json"))

    instructions: str = SURVEY["sections"]["annotation"]["description"]
    samples: list[str] = [
        sample["features"][0]["content"] for sample in SURVEY["samples"]
    ]
    choices: list[str] = SURVEY["annotation"][0]["options"]

    conversations: list[list[dict]] = [
        [
            {"role": "system", "content": instructions},
            {"role": "user", "content": sample},
        ]
        for sample in samples
    ]

    results = {}
    for model_label, model_url in ca.constants.MODELS.items():
        predictions: list[vllm.RequestOutput] = ca.inference.batch_infer(
            conversations=conversations,
            model_name=model_url,
            sampling_params=vllm.SamplingParams(
                structured_outputs=vllm.sampling_params.StructuredOutputsParams(
                    choice=choices
                ),
            ),
        )

        results[model_label] = [item.outputs[0].text for item in predictions]

    pandas.DataFrame.from_dict(results).to_csv(f"data/model/base.{SAMPLE}.csv")
