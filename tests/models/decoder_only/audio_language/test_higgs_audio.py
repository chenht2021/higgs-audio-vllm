# SPDX-License-Identifier: Apache-2.0
# ruff: noqa
import asyncio
import base64
import io
import json
import os
import re
import textwrap
from typing import Any

import jiwer
import librosa
import numpy as np
import pytest
import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from vllm import LLM, SamplingParams
from vllm.entrypoints.bosonai.serving_audio import HiggsAudioServingAudio
from vllm.entrypoints.bosonai.serving_chat import HiggsAudioServingChat
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (AudioSpeechRequest,
                                              ChatCompletionRequest)
from vllm.entrypoints.openai.serving_models import (BaseModelPath,
                                                    OpenAIServingModels)
from vllm.model_executor.models.higgs_audio_tokenizer import (
    AudioTokenizer, revert_delay_pattern)
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.async_llm import AsyncEngineArgs, AsyncLLM

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
    # fmt: off
    "{% set loop_messages = messages %}"
    "{% for message in loop_messages %}"
    "{% set content = '<|start_header_id|>' + message['role'] + "
    "'<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}"
    "{% if loop.index0 == 0 %}"
    "{% set content = bos_token + content %}"
    "{% endif %}"
    "{% if message['role'] == 'assistant' and '<|audio_bos|><|AUDIO|>' in message['content'] %}"
    "{% set content = content.replace('<|audio_bos|><|AUDIO|>', '<|audio_out_bos|><|AUDIO|>') %}"
    "{% endif %}"
    "{{ content }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n<|audio_out_bos|>' }}"
    "{% endif %}")
# fmt: on

OPENAI_TTS_SAMPLE_RATE = 24000


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


def prepare_tts_voice_clone_sample(text: str, ref_audio_path: str):
    ref_audio_base64 = encode_base64_content_from_file(ref_audio_path)
    tts_sample = [{
        "role":
        "system",
        "content":
        "Convert the following text from the user into speech."
    }, {
        "role":
        "user",
        "content":
        "The device would work during the day as well, if you took steps to either block direct sunlightor point it away from the sun."
    }, {
        "role":
        "assistant",
        "content": [{
            "type": "input_audio",
            "input_audio": {
                "data": ref_audio_base64,
                "format": "wav",
            }
        }]
    }, {
        "role": "user",
        "content": text
    }]
    return tts_sample


def prepare_emergent_tts_sample(text: str):
    tts_sample = [{
        "role":
        "system",
        "content":
        "You are an AI assistant designed to convert text into speech for podcast.\nIf the user's message includes a [SPEAKER*] tag, do not read out the tag and generate speech for the following text, using the specified voice. If no speaker tag is present, select a suitable voice for the audiobook's tone and style.\nThe audio may contain artifacts."
    }, {
        "role": "user",
        "content": text
    }]
    return tts_sample


@pytest.fixture(scope="module")
def tts_voice_clone_sample_1():
    return prepare_tts_voice_clone_sample(
        "Mr. Bounce was very small, and like a rubber ball.", "en_woman_1.wav")


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
    return re.sub(r'[^\w\s]', ' ', s)


def clean_speaker_tag(s: str) -> str:
    # Remove the speaker tag like [SPEAKER*] from the string
    return re.sub(r"\[SPEAKER\d+\] ", "", s)


def remove_newlines(s: str) -> str:
    # Remove all newlines from the string
    return s.replace("\n", " ")


@pytest.mark.parametrize("model_name, tokenizer",
                         [("bosonai/higgs-audio-v2-generation-3B-base",
                           "bosonai/higgs-audio-v2-tokenizer")])
def test_audio_tts_zero_shot(speech_samples, asr_pipeline, model_name,
                             tokenizer):
    torch.random.manual_seed(0)
    np.random.seed(0)

    batch_size = 20
    conversations = [
        prepare_zero_shot_conversation(speech_samples[i])
        for i in range(batch_size)
    ]
    llm = LLM(model=model_name, max_model_len=1024, gpu_memory_utilization=0.6)
    sampling_params = SamplingParams(
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        max_tokens=500,
        seed=0,
        stop=["<|eot_id|>", "<|end_of_text|>", "<|audio_eos|>"])
    outputs = llm.chat(
        conversations,
        sampling_params=sampling_params,
        use_tqdm=False,
        chat_template=AUDIO_OUT_CHAT_TEMPLATE,
    )

    audio_tokenizer = AudioTokenizer(model=tokenizer)

    reference = ""
    hypothesis = ""
    for i in range(len(outputs)):
        audio_out_ids = \
            np.array(outputs[i].outputs[0].mm_token_ids)[1:-1].transpose(1, 0).clip(0, audio_tokenizer.codebook_size - 1)
        reverted_audio_out_ids = revert_delay_pattern(audio_out_ids)
        decoded_audio, sr = audio_tokenizer.decode(reverted_audio_out_ids)
        asr_text = _get_asr(decoded_audio, sr, asr_pipeline)
        # sf.write(f"audio_dumps/audio_out_{i}.wav", decoded_audio, sr)
        reference += clean_punctuation(speech_samples[i]).lower()
        hypothesis += clean_punctuation(asr_text).lower()

    wer = jiwer.wer(reference, hypothesis)
    print(f"WER: {wer}")
    assert wer < 0.1


@pytest.mark.parametrize("model_path, audio_tokenizer",
                         [("bosonai/higgs-audio-v2-generation-3B-base",
                           "bosonai/higgs-audio-v2-tokenizer")])
def test_audio_tts_voice_clone(speech_samples, asr_pipeline, model_path,
                               audio_tokenizer):
    torch.random.manual_seed(0)
    np.random.seed(0)

    os.environ["HIGGS_AUDIO_TOKENIZER"] = audio_tokenizer
    audio_tokenizer = AudioTokenizer(model=audio_tokenizer)

    batch_size = 20
    ref_audio_paths = ["en_woman_1.wav", "en_man_1.wav"]
    conversations = [
        prepare_tts_voice_clone_sample(
            speech_samples[i], ref_audio_paths[i % len(ref_audio_paths)])
        for i in range(batch_size)
    ]
    llm = LLM(model=model_path, max_model_len=1024)
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=500,
        stop=["<|eot_id|>", "<|end_of_text|>", "<|audio_eos|>"])

    outputs = llm.chat(
        conversations,
        sampling_params=sampling_params,
        use_tqdm=False,
        chat_template=AUDIO_OUT_CHAT_TEMPLATE,
    )

    reference = ""
    hypothesis = ""
    for i in range(len(outputs)):
        audio_out_ids = \
            np.array(outputs[i].outputs[0].mm_token_ids).transpose(1, 0).clip(0, audio_tokenizer.codebook_size - 1)
        reverted_audio_out_ids = revert_delay_pattern(audio_out_ids)
        decoded_audio, sr = audio_tokenizer.decode(reverted_audio_out_ids)
        asr_text = _get_asr(decoded_audio, sr, asr_pipeline)
        sf.write(f"audio_dumps/audio_out_{i}.wav", decoded_audio, sr)
        reference += clean_punctuation(speech_samples[i]).lower()
        hypothesis += clean_punctuation(asr_text).lower()

    wer = jiwer.wer(reference, hypothesis)
    print(f"WER: {wer}")
    assert wer < 0.05


@pytest.fixture(scope="module")
def dialogue_sample_1():
    audio_woman_path = "en_woman_1.wav"
    audio_woman_base64 = encode_base64_content_from_file(audio_woman_path)
    audio_man_path = "en_man_1.wav"
    audio_man_base64 = encode_base64_content_from_file(audio_man_path)
    dialogue_sample = [{
        "role":
        "user",
        "content": [
            "[SPEAKER0] The device would work during the day as well, if you took steps to either block direct sunlightor point it away from the sun."
        ]
    }, {
        "role":
        "assistant",
        "content": [{
            "type": "input_audio",
            "input_audio": {
                "data": audio_woman_base64,
            }
        }]
    }, {
        "role":
        "user",
        "content": [
            "[SPEAKER1] Maintaining your ability to learn translates into increased marketability, improved career optionsand higher salaries."
        ]
    }, {
        "role":
        "assistant",
        "content": [{
            "type": "input_audio",
            "input_audio": {
                "data": audio_man_base64,
            }
        }]
    }, {
        "role":
        "user",
        "content": [
            "[SPEAKER0] Hello, how are you doing today?\n[SPEAKER1] I'm doing great, thank you!"
        ]
    }]
    return dialogue_sample


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name, tokenizer",
                         [("bosonai/higgs-audio-v2-generation-3B-base",
                           "bosonai/higgs-audio-v2-tokenizer")])
async def test_audio_tts_voice_clone_async(speech_samples, asr_pipeline,
                                           model_name, tokenizer):
    torch.random.manual_seed(0)
    np.random.seed(0)

    os.environ["HIGGS_AUDIO_TOKENIZER"] = tokenizer
    model_path = model_name

    audio_tokenizer = AudioTokenizer(model=tokenizer)

    batch_size = 20
    ref_audio_paths = ["en_woman_1.wav", "en_man_1.wav"]
    conversations = [
        prepare_tts_voice_clone_sample(
            speech_samples[i], ref_audio_paths[i % len(ref_audio_paths)])
        for i in range(batch_size)
    ]

    vllm_config = AsyncEngineArgs(model=model_path,
                                  max_model_len=1024,
                                  limit_mm_per_prompt={
                                      "audio": 50
                                  }).create_engine_config(
                                      UsageContext.ENGINE_CONTEXT)
    engine = AsyncLLM.from_vllm_config(vllm_config)
    model_config = await engine.get_model_config()
    base_model_paths = [
        BaseModelPath(name="higgs_audio", model_path=model_path)
    ]
    openai_serving_models = OpenAIServingModels(
        engine_client=engine,
        model_config=model_config,
        base_model_paths=base_model_paths,
    )
    serving_chat = HiggsAudioServingChat(
        engine,
        model_config,
        openai_serving_models,
        response_role="assistant",
        request_logger=None,
        chat_template_content_format="auto",
        audio_tokenizer=audio_tokenizer,
    )

    async def process_request(conversation):
        request_json = ChatCompletionRequest(
            messages=conversation,
            model="higgs_audio",
            max_completion_tokens=500,
            temperature=0.7,
            stop=["<|eot_id|>", "<|end_of_text|>", "<|audio_eos|>"],
            modalities=["audio", "text"])
        output = await serving_chat.create_chat_completion(request_json)
        return output

    # Create tasks for each conversation
    tasks = []
    for i in range(batch_size):
        task = asyncio.create_task(process_request(conversations[i]))
        tasks.append(task)

        # Add a 10ms delay between requests
        await asyncio.sleep(0.01)

    outputs = await asyncio.gather(*tasks)
    assert len(outputs) == batch_size
    reference = ""
    hypothesis = ""
    for i, output in enumerate(outputs):
        audio_data = base64.b64decode(output.choices[0].message.audio.data)
        # with open(f"audio_dumps/audio_out_{i}.wav", "wb") as f:
        #     f.write(audio_data)
        audio_stream = io.BytesIO(audio_data)
        decoded_audio, sr = sf.read(audio_stream, dtype='int16')
        asr_text = _get_asr(decoded_audio.astype(np.float32), sr, asr_pipeline)
        reference += clean_punctuation(speech_samples[i % 20]).lower()
        hypothesis += clean_punctuation(asr_text).lower()

    wer = jiwer.wer(reference, hypothesis)
    print(f"WER: {wer}")
    assert wer < 0.05


@pytest.mark.asyncio
async def test_audio_tts_voice_clone_async_streaming(speech_samples,
                                                     asr_pipeline):
    torch.random.manual_seed(0)
    np.random.seed(0)

    model_path = "bosonai/higgs-audio-v2-generation-3B-base"
    audio_tokenizer_path = "bosonai/higgs-audio-v2-tokenizer"

    os.environ["HIGGS_AUDIO_TOKENIZER"] = audio_tokenizer_path
    audio_tokenizer = AudioTokenizer(model=audio_tokenizer_path, device="cuda")

    batch_size = 20
    conversations = [
        prepare_emergent_tts_sample(speech_samples[i])
        for i in range(batch_size)
    ]

    vllm_config = AsyncEngineArgs(model=model_path,
                                  max_model_len=1024,
                                  limit_mm_per_prompt={
                                      "audio": 50
                                  }).create_engine_config(
                                      UsageContext.ENGINE_CONTEXT)
    engine = AsyncLLM.from_vllm_config(vllm_config)
    model_config = await engine.get_model_config()
    base_model_paths = [
        BaseModelPath(name="higgs_audio", model_path=model_path)
    ]
    openai_serving_models = OpenAIServingModels(
        engine_client=engine,
        model_config=model_config,
        base_model_paths=base_model_paths,
    )
    serving_chat = HiggsAudioServingChat(
        engine,
        model_config,
        openai_serving_models,
        response_role="assistant",
        request_logger=None,
        chat_template_content_format="auto",
        audio_tokenizer=audio_tokenizer,
    )

    async def process_request(conversation):
        request_json = ChatCompletionRequest(
            messages=conversation,
            model="higgs_audio",
            max_completion_tokens=500,
            top_p=0.95,
            top_k=50,
            temperature=1,
            stop=["<|eot_id|>", "<|end_of_text|>", "<|audio_eos|>"],
            stream=True,
            modalities=["audio", "text"])
        audio_bytes_io = io.BytesIO()
        i = 0
        output = await serving_chat.create_chat_completion(request_json)
        async for chunk_str in output:
            # Parse the SSE format: "data: {json}\n\n"
            if chunk_str.startswith("data: "):
                json_str = chunk_str[6:]  # Remove "data: " prefix
                if json_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(json_str)
                    # print(f"Chunk {i}: {chunk}")
                    if (chunk.get("choices")
                            and chunk["choices"][0].get("delta")
                            and chunk["choices"][0]["delta"].get("audio")):
                        audio_bytes = base64.b64decode(
                            chunk["choices"][0]["delta"]["audio"]["data"])
                        audio_bytes_io.write(audio_bytes)
                        i += 1
                except json.JSONDecodeError:
                    # Skip malformed JSON
                    continue
            else:
                print(chunk_str)
        audio_bytes_io.seek(0)
        audio_data = np.frombuffer(audio_bytes_io.getvalue(), dtype=np.int16)
        return audio_data

    # Create tasks for each conversation
    tasks = []
    for i in range(batch_size):
        task = asyncio.create_task(process_request(conversations[i]))
        tasks.append(task)

        # Add a 10ms delay between requests
        await asyncio.sleep(0.01)

    outputs = await asyncio.gather(*tasks)
    assert len(outputs) == batch_size
    reference = ""
    hypothesis = ""
    for i, output in enumerate(outputs):
        sf.write(f"audio_dumps/audio_out_{i}.wav", output,
                 audio_tokenizer.sampling_rate)
        asr_text = _get_asr(output.astype(np.float32),
                            audio_tokenizer.sampling_rate, asr_pipeline)
        reference += clean_punctuation(speech_samples[i % 20]).lower()
        hypothesis += clean_punctuation(asr_text).lower()

    wer = jiwer.wer(reference, hypothesis)
    print(f"WER: {wer}")
    assert wer < 0.05


@pytest.mark.asyncio
async def test_audio_speech_api(speech_samples, asr_pipeline):
    torch.random.manual_seed(0)
    np.random.seed(0)

    model_path = "bosonai/higgs-audio-v2-generation-3B-base"
    audio_tokenizer_path = "bosonai/higgs-audio-v2-tokenizer"
    os.environ["HIGGS_AUDIO_TOKENIZER"] = audio_tokenizer_path
    audio_tokenizer = AudioTokenizer(model=audio_tokenizer_path, device="cuda")

    batch_size = 1
    vllm_config = AsyncEngineArgs(model=model_path,
                                  max_model_len=1024,
                                  limit_mm_per_prompt={
                                      "audio": 50
                                  }).create_engine_config(
                                      UsageContext.ENGINE_CONTEXT)
    engine = AsyncLLM.from_vllm_config(vllm_config)
    model_config = await engine.get_model_config()
    base_model_paths = [
        BaseModelPath(name="higgs_audio", model_path=model_path)
    ]
    openai_serving_models = OpenAIServingModels(
        engine_client=engine,
        model_config=model_config,
        base_model_paths=base_model_paths,
    )
    voice_presets_dir = os.path.join(os.path.dirname(__file__), "..", "..",
                                     "..", "..",
                                     "vllm/entrypoints/bosonai/voice_presets")
    voice_presets = json.load(
        open(os.path.join(voice_presets_dir, "config.json")))

    request_logger = RequestLogger(max_log_len=None)
    serving_audio = HiggsAudioServingAudio(
        engine,
        model_config,
        openai_serving_models,
        request_logger=request_logger,
        chat_template_content_format="auto",
        audio_tokenizer=audio_tokenizer,
        voice_presets_dir=voice_presets_dir,
    )

    async def process_request(text: str):
        request_json = AudioSpeechRequest(
            voice="en_woman_1",
            input=text,
            model="higgs_audio",
            response_format="pcm",
        )
        output = await serving_audio.create_audio_speech_stream(
            request=request_json,
            voice_presets=voice_presets,
        )
        audio_bytes = b''
        async for chunk in output:
            audio_bytes += chunk

        audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
        return audio_data

    # Create tasks for each conversation
    tasks = []
    for i in range(batch_size):
        task = asyncio.create_task(process_request(speech_samples[i % 20]))
        tasks.append(task)

        # Add a 10ms delay between requests
        await asyncio.sleep(0.01)

    outputs = await asyncio.gather(*tasks)
    assert len(outputs) == batch_size
    reference = ""
    hypothesis = ""
    for i, output in enumerate(outputs):
        sf.write(f"audio_dumps/audio_out_{i}.wav", output,
                 OPENAI_TTS_SAMPLE_RATE)
        asr_text = _get_asr(output.astype(np.float32), OPENAI_TTS_SAMPLE_RATE,
                            asr_pipeline)
        reference += clean_punctuation(speech_samples[i % 20]).lower()
        hypothesis += clean_punctuation(asr_text).lower()

    wer = jiwer.wer(reference, hypothesis)
    print(f"WER: {wer}")
    assert wer < 0.05
