# SPDX-License-Identifier: Apache-2.0
# ruff: noqa
import base64
import os
import re
from typing import Any

import librosa
import numpy as np
import pytest
import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from vllm import LLM, SamplingParams
from vllm.model_executor.models.higgs_audio_tokenizer import (
    AudioTokenizer, revert_delay_pattern)

TEXT_OUT_CHAT_TEMPLATE = (
    "{% set loop_messages = messages %}"
    "{% for message in loop_messages %}"
    "{% set content = '<|start_header_id|>' + message['role'] + "
    "'<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}"
    "{% if loop.index0 == 0 %}"
    "{% set content = bos_token + content %}"
    "{% endif %}"
    "{{ content }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    "{% endif %}")

AUDIO_OUT_CHAT_TEMPLATE = (
    "{% set loop_messages = messages %}"
    "{% for message in loop_messages %}"
    "{% set content = '<|start_header_id|>' + message['role'] + "
    "'<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}"
    "{% if loop.index0 == 0 %}"
    "{% set content = bos_token + content %}"
    "{% endif %}"
    "{{ content }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n<|audio_out_bos|><|AUDIO_OUT|>' }}"
    "{% endif %}")

TEST_MODEL_PATH = "/fsx/models/higgs_audio_test_models"


@pytest.fixture(scope="module")
def speech_samples():
    with open("speech_samples.txt", "r") as f:
        return [line.strip() for line in f.readlines()]


def prepare_zero_shot_conversation(text: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": "Convert the following text from the user into speech."
        },
        {
            "role": "user",
            "content": text,
        },
    ]


@pytest.fixture(scope="module")
def asr_pipeline():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3-turbo"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id,
                                                      torch_dtype=torch_dtype,
                                                      low_cpu_mem_usage=True,
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
    return pipe


def _get_asr(wv, sr, pipe):
    data = librosa.resample(wv, orig_sr=sr, target_sr=16000)
    result = pipe(data)
    return result['text'].strip()


def encode_base64_content_from_file(file_path: str) -> str:
    """Encode a content from a local file to base64 format using librosa."""
    # Read the MP3 file as binary and encode it directly to Base64
    with open(file_path, "rb") as audio_file:
        audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")
    return audio_base64


def clean_punctuation(s):
    return re.sub(r'[^\w\s]', '', s)


def test_text_in_audio_out_zero_shot(speech_samples, asr_pipeline):
    batch_size = 5
    conversations = [
        prepare_zero_shot_conversation(speech_samples[i])
        for i in range(batch_size)
    ]
    model_path = os.path.join(TEST_MODEL_PATH, "higgs_audio_tts_1b_20250325")
    llm = LLM(model=model_path, max_model_len=1024)
    sampling_params = SamplingParams(temperature=0.7,
                                     max_tokens=500,
                                     stop=["<|eot_id|>", "<|end_of_text|>"])

    outputs = llm.chat(
        conversations,
        sampling_params=sampling_params,
        use_tqdm=False,
        chat_template=AUDIO_OUT_CHAT_TEMPLATE,
    )

    audio_tokenizer = AudioTokenizer(
        "xcodec_tps25_0215",
        downloaded_model_path=
        "/fsx/models/higgs_audio_test_models/xcodec_tps25_0215/")
    for i in range(len(outputs)):
        audio_out_ids = \
            np.array(outputs[i].outputs[0].mm_token_ids).transpose(1, 0).clip(0, audio_tokenizer.codebook_size - 1)
        reverted_audio_out_ids = revert_delay_pattern(audio_out_ids)
        decoded_audio, sr = audio_tokenizer.decode(reverted_audio_out_ids)
        asr_text = _get_asr(decoded_audio, sr, asr_pipeline)
        # sf.write(f"audio_out_{i}.wav", decoded_audio, sr)
        assert clean_punctuation(
            speech_samples[i]).lower() == clean_punctuation(asr_text).lower()


def test_audio_in_text_out():
    model_path = os.path.join(TEST_MODEL_PATH, "higgs_audio_in_3b_20241222")
    llm = LLM(model=model_path, max_model_len=1024)
    audio_path = "./audiobook_sample.mp3"
    audio_base64 = encode_base64_content_from_file(audio_path)
    file_suffix = audio_path.split(".")[-1]
    conversation = [
        {
            "role": "system",
            "content": "Transcribe the audio.",
        },
        {
            "role":
            "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {
                        # Any format supported by librosa is supported
                        "url":
                        f"data:audio/{file_suffix};base64,{audio_base64}"
                    },
                },
            ],
        },
    ]
    sampling_params = SamplingParams(temperature=0,
                                     stop=["<|eot_id|>", "<|end_of_text|>"],
                                     max_tokens=512)
    outputs = llm.chat(
        conversation,
        sampling_params=sampling_params,
        use_tqdm=False,
        chat_template=TEXT_OUT_CHAT_TEMPLATE,
    )
    reference_output = (
        "The spookie just tells you how you're doing up. He is a great ex-plover. "
        "No grass left, no cajun, there's lots of grass left, there'll be lots of "
        "grass left and just a few some for the rabbits. What rabbit?")
    assert outputs[0].outputs[0].text == reference_output
