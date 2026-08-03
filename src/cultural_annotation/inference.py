import rich
import rich.progress
import vllm


def chunk_list(data: list[any], chunk_size: int) -> list[list[any]]:
    num_chunks = len(data) // chunk_size + 1
    return [data[i * chunk_size : (i + 1) * chunk_size] for i in range(num_chunks)]


def batch_infer(
    conversations: list[list[dict]],
    model_name: str,
    sampling_params: vllm.SamplingParams,
    batch_size: int = 32,
    llm_kwargs: dict | None = None,
) -> list[vllm.RequestOutput]:
    if llm_kwargs is None:
        llm_kwargs = {"gpu_memory_utilization": 0.85, "max_model_len": 5_000}

    rich.print(f"> initializing model: {model_name}")
    model: vllm.LLM = vllm.LLM(model=model_name, **llm_kwargs)

    rich.print("> performing inference")
    predictions: list[vllm.RequestOutput] = []
    for batch in rich.progress.track(
        chunk_list(conversations, batch_size),
        description="processing batches",
    ):
        predictions.extend(
            model.chat(
                batch,
                sampling_params,
                use_tqdm=False,
                chat_template_kwargs={"enable_thinking": False},
            )
        )

    return predictions
