#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark speech API throughput using streaming."""

import asyncio
import logging
import os
import random
import re
import time
import traceback
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from logging import getLogger

import click
import jiwer
import librosa
import numpy as np
import openai
import soundfile as sf
import torch
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Configure logging to display on terminal
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()])
logger = getLogger(__name__)

# Disable httpx INFO logs
logging.getLogger("httpx").setLevel(logging.WARNING)

OPENAI_TTS_SAMPLE_RATE = 24000


@dataclass
class FetchResult:
    text: str
    audio_bytes: bytes
    first_chunk_latency: float
    total_latency: float
    max_iter_chunk_latency: float = float("nan")


def load_text_samples(file_path: str) -> list[str]:
    """Load text samples from a file."""
    samples = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                samples.append(line)

    logger.info("Loaded %d text samples from %s", len(samples), file_path)
    return samples


async def get_request(
    text_samples: list[str],
    num_requests: int,
    request_rate: float,
) -> AsyncGenerator[str, None]:
    """Generate speech synthesis requests at the specified rate."""
    for _ in range(num_requests):
        # Randomly select a text sample
        text = random.choice(text_samples)
        yield text

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        # Sample the request interval from the exponential distribution
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval
        await asyncio.sleep(interval)


async def send_speech_request(
    api_url: str,
    api_key: str,
    model: str,
    text: str,
    request_id: int,
) -> tuple[str, bytes, float, float]:
    """Send a request to the speech API and return timing metrics."""
    client = openai.AsyncOpenAI(api_key=api_key, base_url=api_url)

    request_start_time = time.perf_counter()
    first_chunk_time = None
    audio_bytes = b""

    params = {
        "model": model,
        "input": text,
        "voice": "default",
        "response_format": "pcm",
        "speed": 1.0,
        "extra_body": {
            "temperature": 1.0,
            "max_tokens": 2048,
        },
    }

    max_iter_chunk_latency = 0
    last_chunk_arrival_time = None
    try:
        async with client.audio.speech.with_streaming_response.create(
                **params) as response:
            async for chunk in response.iter_bytes(chunk_size=1024):
                if first_chunk_time is None:
                    first_chunk_time = time.perf_counter()
                audio_bytes += chunk
                if last_chunk_arrival_time is not None:
                    iter_chunk_latency = time.perf_counter(
                    ) - last_chunk_arrival_time
                    max_iter_chunk_latency = max(max_iter_chunk_latency,
                                                 iter_chunk_latency)
                last_chunk_arrival_time = time.perf_counter()

    except Exception as e:
        logger.error("Error processing request: %s", e)
        logger.error("Traceback: %s", traceback.format_exc())
        return FetchResult(text, b"", 0, 0)

    request_end_time = time.perf_counter()

    # Calculate metrics
    total_latency = request_end_time - request_start_time
    first_chunk_latency = (first_chunk_time -
                           request_start_time if first_chunk_time else 0)

    return FetchResult(text, audio_bytes, first_chunk_latency, total_latency,
                       max_iter_chunk_latency)


async def fetch_results(queue: asyncio.Queue) -> list[FetchResult]:
    """Collect results from the queue."""
    results = []
    try:
        while True:
            task = await queue.get()
            results.append(await task)
            queue.task_done()
    except asyncio.CancelledError:
        pass
    return results


async def benchmark(
    api_url: str,
    api_key: str,
    model: str,
    text_samples: list[str],
    num_requests: int,
    request_rate: float,
) -> list[FetchResult]:
    """Run the benchmark with the specified parameters."""
    if request_rate == 0:
        # When request rate is 0, send requests sequentially
        logger.info("Starting benchmark with %d samples sequentially...",
                    num_requests)
        results = []
        for idx in tqdm(range(num_requests), total=num_requests):
            text = random.choice(text_samples)
            results.append(await send_speech_request(api_url, api_key, model,
                                                     text, idx))
    else:
        # Send requests at the specified rate
        logger.info("Starting benchmark with %d samples at %.2f requests/s...",
                    num_requests, request_rate)
        queue = asyncio.Queue()
        fetch_task_result = asyncio.create_task(fetch_results(queue))

        count = 0
        request_gen = get_request(text_samples, num_requests, request_rate)
        async for text in async_tqdm(request_gen, total=num_requests):
            task = asyncio.create_task(
                send_speech_request(api_url, api_key, model, text, count))
            queue.put_nowait(task)
            count += 1

        await queue.join()
        fetch_task_result.cancel()
        results = await fetch_task_result

    return results


def clean_punctuation(s: str) -> str:
    return re.sub(r'[^\w\s]', ' ', s)


def check_audio_results(results: list[FetchResult]):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3-turbo"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id,
                                                      torch_dtype=torch_dtype,
                                                      use_safetensors=True)
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    avg_wer = 0.0

    for i, result in enumerate(results):
        audio_bytes = result.audio_bytes
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        try:
            audio_array = librosa.resample(audio_array.astype(np.float32),
                                           orig_sr=OPENAI_TTS_SAMPLE_RATE,
                                           target_sr=16000)
            transcription = pipe(audio_array)["text"].strip()
            logger.info("req %d: %s", i, transcription)
            # Calculate the word error rate
            word_error_rate = jiwer.wer(clean_punctuation(transcription),
                                        clean_punctuation(result.text))
            logger.info("req %d: word error rate: %.5f", i, word_error_rate)
            avg_wer += word_error_rate
        except Exception as e:
            logger.error("Error processing request %d: %s", i, e)
            logger.error("Traceback: %s", traceback.format_exc())
            continue

    avg_wer /= len(results)
    logger.info("Average word error rate: %.5f", avg_wer)


def save_audio_results(results: list[FetchResult]):
    saved_audio_dir = "benchmark_audio_output"
    os.makedirs(saved_audio_dir, exist_ok=True)

    for i, result in enumerate(results):
        audio_bytes = result.audio_bytes
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_file_path = os.path.join(saved_audio_dir, f"req_{i}.wav")
        sf.write(audio_file_path, audio_array, OPENAI_TTS_SAMPLE_RATE)


@click.command()
@click.option("--api-base",
              default="http://0.0.0.0:26003/v1",
              help="The API base URL.")
@click.option("--api-key", default="EMPTY", help="Your API key.")
@click.option("--model", default="higgs-audio", help="The TTS model to use.")
@click.option("--samples-file",
              default="./examples/speech_samples.txt",
              help="Path to the text samples file.")
@click.option(
    "--request-rate",
    type=float,
    default=float("inf"),
    help="Number of requests per second. If this is inf, "
    "then all the requests are sent at time 0. "
    "Otherwise, we use Poisson process to synthesize "
    "the request arrival times.",
)
@click.option(
    "--n-requests",
    type=int,
    default=20,
    help="Number of requests to send.",
)
@click.option("--seed", type=int, default=42, help="Random seed.")
@click.option("--check-results",
              is_flag=True,
              help="Whether to check the results.")
@click.option("--save-audio",
              is_flag=True,
              help="Whether to save the audio results.")
def main(
    api_base,
    api_key,
    model,
    samples_file,
    request_rate,
    n_requests,
    seed,
    check_results,
    save_audio,
):
    """Benchmark the speech API throughput."""
    random.seed(seed)
    np.random.seed(seed)

    # Load text samples
    text_samples = load_text_samples(samples_file)
    if not text_samples:
        logger.error("No text samples found in %s", samples_file)
        return

    # Run the benchmark
    benchmark_start_time = time.perf_counter()
    results = asyncio.run(
        benchmark(api_base, api_key, model, text_samples, n_requests,
                  request_rate))

    benchmark_end_time = time.perf_counter()
    benchmark_time = benchmark_end_time - benchmark_start_time

    # Throughput metrics
    logger.info("Total benchmark time: %.2f s", benchmark_time)
    logger.info("Request throughput: %.2f requests/s",
                n_requests / benchmark_time)

    if save_audio:
        save_audio_results(results)

    if check_results:
        check_audio_results(results)

    # Latency metrics
    latencies = [fetch_result.total_latency for fetch_result in results]
    # Sort once for efficient percentile calculations
    sorted_latencies = sorted(latencies)
    p50_latency = sorted_latencies[int(len(sorted_latencies) * 0.5)]
    p90_latency = sorted_latencies[int(len(sorted_latencies) * 0.9)]

    logger.info("Request latency: mean=%.2f s, std=%.2f s", np.mean(latencies),
                np.std(latencies))
    logger.info("Request latency: p50=%.2f s, p90=%.2f s", p50_latency,
                p90_latency)

    first_chunk_latencies = [
        fetch_result.first_chunk_latency for fetch_result in results
    ]
    # Sort once for efficient percentile calculations
    sorted_first_chunk_latencies = sorted(first_chunk_latencies)
    p50_first_chunk = sorted_first_chunk_latencies[int(
        len(sorted_first_chunk_latencies) * 0.5)]
    p90_first_chunk = sorted_first_chunk_latencies[int(
        len(sorted_first_chunk_latencies) * 0.9)]

    logger.info(
        "First chunk latency: mean=%.2f s, std=%.2f s",
        np.mean(first_chunk_latencies),
        np.std(first_chunk_latencies),
    )
    logger.info(
        "First chunk latency: p50=%.2f s, p90=%.2f s",
        p50_first_chunk,
        p90_first_chunk,
    )

    max_iter_chunk_latencies = [
        fetch_result.max_iter_chunk_latency for fetch_result in results
    ]
    # Sort once for efficient percentile calculations
    sorted_max_iter_chunk_latencies = sorted(max_iter_chunk_latencies)
    p50_max_iter_chunk = sorted_max_iter_chunk_latencies[int(
        len(sorted_max_iter_chunk_latencies) * 0.5)]
    p90_max_iter_chunk = sorted_max_iter_chunk_latencies[int(
        len(sorted_max_iter_chunk_latencies) * 0.9)]

    logger.info(
        "Max iter chunk latency: mean=%.2f s, std=%.2f s",
        np.mean(max_iter_chunk_latencies),
        np.std(max_iter_chunk_latencies),
    )
    logger.info(
        "Max iter chunk latency: p50=%.2f s, p90=%.2f s",
        p50_max_iter_chunk,
        p90_max_iter_chunk,
    )


if __name__ == "__main__":
    main()
