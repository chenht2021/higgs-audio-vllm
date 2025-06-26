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

TEST_MODEL_PATH = "/fsx/models/higgs_audio_test_models"

OPENAI_TTS_SAMPLE_RATE = 24000


def test_tts_chat_template():
    from transformers import AutoTokenizer

    chat_template = AUDIO_OUT_CHAT_TEMPLATE
    model_path = os.path.join(TEST_MODEL_PATH, "higgs_audio_tts_1b_20250325")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    conversation = [{
        "role": "user",
        "content": "Hello1"
    }, {
        "role": "assistant",
        "content": "<|audio_bos|><|AUDIO|>"
    }, {
        "role": "user",
        "content": "Hello2"
    }]

    result = tokenizer.apply_chat_template(
        chat_template=chat_template,
        conversation=conversation,
        tokenize=False,
        add_generation_prompt=True,
    )

    ref = textwrap.dedent("""
        <|begin_of_text|><|start_header_id|>user<|end_header_id|>

        Hello1<|eot_id|><|start_header_id|>assistant<|end_header_id|>

        <|audio_out_bos|><|AUDIO|><|eot_id|><|start_header_id|>user<|end_header_id|>

        Hello2<|eot_id|><|start_header_id|>assistant<|end_header_id|>

        <|audio_out_bos|><|AUDIO_OUT|>
    """).lstrip('\n').rstrip('\n')
    assert result == ref


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


@pytest.mark.parametrize("model_name", [
    "higgs_audio_tts_1b_20250325",
    "higgs_audio_dual_ffn_1b_20250513",
])
def test_audio_tts_zero_shot(speech_samples, asr_pipeline, model_name):
    torch.random.manual_seed(0)
    np.random.seed(0)

    batch_size = 20
    conversations = [
        prepare_zero_shot_conversation(speech_samples[i])
        for i in range(batch_size)
    ]
    model_path = os.path.join(TEST_MODEL_PATH, model_name)
    llm = LLM(model=model_path, max_model_len=1024)
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

    if model_name == "higgs_audio_dual_ffn_1b_20250513":
        audio_tokenizer = AudioTokenizer("xcodec_0507_exp_1",
                                         downloaded_model_path=os.path.join(
                                             TEST_MODEL_PATH,
                                             "xcodec_tps50_0507_exp1"))
    else:
        audio_tokenizer = AudioTokenizer("xcodec_tps25_0215",
                                         downloaded_model_path=os.path.join(
                                             TEST_MODEL_PATH,
                                             "xcodec_tps25_0215"))

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


def test_audio_tts_voice_clone(speech_samples, asr_pipeline):
    torch.random.manual_seed(0)
    np.random.seed(0)

    audio_tokenizer_type = "xcodec_tps25_0215"
    audio_tokenizer_path = "/fsx/models/higgs_audio_test_models/xcodec_tps25_0215/"
    os.environ["HIGGS_AUDIO_TOKENIZER"] = audio_tokenizer_type
    os.environ["HIGGS_AUDIO_TOKENIZER_PATH"] = audio_tokenizer_path

    batch_size = 20
    ref_audio_paths = ["en_woman_1.wav", "en_man_1.wav"]
    conversations = [
        prepare_tts_voice_clone_sample(
            speech_samples[i], ref_audio_paths[i % len(ref_audio_paths)])
        for i in range(batch_size)
    ]
    model_path = os.path.join(TEST_MODEL_PATH, "higgs_audio_tts_1b_20250325")
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

    audio_tokenizer = AudioTokenizer(
        audio_tokenizer_type, downloaded_model_path=audio_tokenizer_path)

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


def test_audio_tts_dialogue(speech_samples, dialogue_sample_1, asr_pipeline):
    audio_tokenizer_type = "xcodec_tps25_0215"
    audio_tokenizer_path = "/fsx/models/higgs_audio_test_models/xcodec_tps25_0215/"
    os.environ["HIGGS_AUDIO_TOKENIZER"] = audio_tokenizer_type
    os.environ["HIGGS_AUDIO_TOKENIZER_PATH"] = audio_tokenizer_path
    model_path = os.path.join(TEST_MODEL_PATH, "higgs_audio_tts_1b_20250325")
    llm = LLM(model=model_path,
              max_model_len=1024,
              limit_mm_per_prompt={"audio": 50})
    sampling_params = SamplingParams(
        temperature=0.7,
        stop=["<|eot_id|>", "<|end_of_text|>", "<|audio_eos|>"],
        max_tokens=512)

    batch_size = 20
    conversations = []
    # Mix the dialogue sample with the voice clone sample
    for i in range(batch_size):
        if i % 2 == 0:
            conversations.append(dialogue_sample_1)
        else:
            conversations.append(
                prepare_tts_voice_clone_sample(speech_samples[i % 20],
                                               "en_woman_1.wav"))

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

    reference = ""
    hypothesis = ""
    for i in range(len(outputs)):
        audio_out_ids = \
            np.array(outputs[i].outputs[0].mm_token_ids).transpose(1, 0).clip(0, audio_tokenizer.codebook_size - 1)
        reverted_audio_out_ids = revert_delay_pattern(audio_out_ids)
        decoded_audio, sr = audio_tokenizer.decode(reverted_audio_out_ids)
        asr_text = _get_asr(decoded_audio, sr, asr_pipeline)
        #sf.write(f"audio_dumps/audio_out_{i}.wav", decoded_audio, sr)
        if i % 2 == 0:
            reference += clean_punctuation(
                remove_newlines(
                    clean_speaker_tag(
                        dialogue_sample_1[-1]["content"][0]))).lower()
        else:
            reference += clean_punctuation(speech_samples[i % 20]).lower()
        hypothesis += clean_punctuation(asr_text).lower()

    wer = jiwer.wer(reference, hypothesis)
    print(f"WER: {wer}")
    print(f"Reference: {reference}")
    print(f"Hypothesis: {hypothesis}")
    assert wer < 0.05


def test_audio_in_text_out():
    os.environ["HIGGS_AUDIO_TOKENIZER"] = "openai/whisper-large-v3-turbo"
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
    sampling_params = SamplingParams(
        temperature=0,
        stop=["<|eot_id|>", "<|end_of_text|>", "<|audio_eos|>"],
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


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [
    "higgs_audio_tts_1b_20250325",
    "higgs_audio_dual_ffn_1b_20250513",
])
async def test_audio_tts_voice_clone_async(speech_samples, asr_pipeline,
                                           model_name):
    torch.random.manual_seed(0)
    np.random.seed(0)
    if model_name == "higgs_audio_dual_ffn_1b_20250513":
        audio_tokenizer_type = "xcodec_0507_exp_1"
        audio_tokenizer_path = os.path.join(TEST_MODEL_PATH,
                                            "xcodec_tps50_0507_exp1")
    else:
        audio_tokenizer_type = "xcodec_tps25_0215"
        audio_tokenizer_path = os.path.join(TEST_MODEL_PATH,
                                            "xcodec_tps25_0215")
    os.environ["HIGGS_AUDIO_TOKENIZER"] = audio_tokenizer_type
    os.environ["HIGGS_AUDIO_TOKENIZER_PATH"] = audio_tokenizer_path
    model_path = os.path.join(TEST_MODEL_PATH, model_name)

    audio_tokenizer = AudioTokenizer(
        audio_tokenizer_type, downloaded_model_path=audio_tokenizer_path)

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
                                  },
                                  enforce_eager=True).create_engine_config(
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

    audio_tokenizer_type = "xcodec_tps25_0215"
    audio_tokenizer_path = "/fsx/models/higgs_audio_test_models/xcodec_tps25_0215/"
    os.environ["HIGGS_AUDIO_TOKENIZER"] = audio_tokenizer_type
    os.environ["HIGGS_AUDIO_TOKENIZER_PATH"] = audio_tokenizer_path
    model_path = "/fsx/models/releases/higgs-audio-generation-3b-v1.0-sft-20250331/"

    audio_tokenizer = AudioTokenizer(
        "xcodec_tps25_0215",
        downloaded_model_path=
        "/fsx/models/higgs_audio_test_models/xcodec_tps25_0215/")

    batch_size = 1
    conversations = [
        prepare_emergent_tts_sample(speech_samples[i])
        for i in range(batch_size)
    ]

    vllm_config = AsyncEngineArgs(model=model_path,
                                  max_model_len=1024,
                                  limit_mm_per_prompt={
                                      "audio": 50
                                  },
                                  enforce_eager=True).create_engine_config(
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

    audio_tokenizer_type = "xcodec_tps25_0215"
    audio_tokenizer_path = "/fsx/models/higgs_audio_test_models/xcodec_tps25_0215/"
    os.environ["HIGGS_AUDIO_TOKENIZER"] = audio_tokenizer_type
    os.environ["HIGGS_AUDIO_TOKENIZER_PATH"] = audio_tokenizer_path
    # model_path = "/fsx/models/releases/higgs-audio-generation-3b-v1.0-sft-20250331/"
    model_path = os.path.join(TEST_MODEL_PATH, "higgs_audio_tts_1b_20250325")

    audio_tokenizer = AudioTokenizer(
        "xcodec_tps25_0215",
        downloaded_model_path=
        "/fsx/models/higgs_audio_test_models/xcodec_tps25_0215/")

    batch_size = 1
    vllm_config = AsyncEngineArgs(model=model_path,
                                  max_model_len=1024,
                                  limit_mm_per_prompt={
                                      "audio": 50
                                  },
                                  enforce_eager=True).create_engine_config(
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
            temperature=0.7,
            top_p=0.95,
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


def prepare_text_audio_interleave_sample():
    system_prompt = """Generate audio following instruction.
<|scene_desc_start|>
SPEAKER0: vocal fry;moderate pitch;monotone;masculine;young adult;slightly fast
SPEAKER1: masculine;moderate;moderate pitch;monotone;mature
In this scene, a group of adventurers is debating whether to investigate a potentially dangerous situation.
<|scene_desc_end|>"""
    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role":
            "user",
            "content":
            "<|generation_instruction_start|>\nGenerate interleaved transcript and audio that lasts for around 10 seconds.\n<|generation_instruction_end|>",
        },
    ]


def split_interleaved_delayed_audios(audio_data: list[list[int]],
                                     audio_tokenizer: AudioTokenizer):
    separator = [1025] * audio_tokenizer.num_codebooks
    groups = []
    current = []
    for row in audio_data:
        current.append(row)
        if row == separator:
            groups.append(current)
            current = []
    # Don't forget the last group if there's no trailing separator
    if current:
        groups.append(current)

    return groups


@pytest.mark.parametrize("model_name", [
    "higgs_audio_interleave_3b_202505013",
    "higgs_audio_dual_ffn_1b_20250513",
])
def test_audio_text_audio_interleave(model_name):
    torch.random.manual_seed(0)
    np.random.seed(0)

    audio_tokenizer_type = "xcodec_0507_exp_1"
    audio_tokenizer_path = "/fsx/models/higgs_audio_test_models/xcodec_tps50_0507_exp1/"
    os.environ["HIGGS_AUDIO_TOKENIZER"] = audio_tokenizer_type
    os.environ["HIGGS_AUDIO_TOKENIZER_PATH"] = audio_tokenizer_path

    batch_size = 1
    conversations = [
        prepare_text_audio_interleave_sample() for _ in range(batch_size)
    ]

    model_path = os.path.join(TEST_MODEL_PATH, model_name)
    llm = LLM(model=model_path, max_model_len=2048)
    sampling_params = SamplingParams(temperature=1.0,
                                     max_tokens=1024,
                                     top_p=0.95,
                                     top_k=50,
                                     stop=["<|eot_id|>", "<|end_of_text|>"])

    outputs = llm.chat(
        conversations,
        sampling_params=sampling_params,
        use_tqdm=False,
        chat_template=TEXT_OUT_CHAT_TEMPLATE,
    )

    audio_tokenizer = AudioTokenizer(
        audio_tokenizer_type, downloaded_model_path=audio_tokenizer_path)

    for i in range(len(outputs)):
        audio_datas = split_interleaved_delayed_audios(
            outputs[i].outputs[0].mm_token_ids, audio_tokenizer)
        decoded_audios = []
        for audio_data in audio_datas:
            audio_data = np.array(audio_data).transpose(1, 0).clip(
                0, audio_tokenizer.codebook_size - 1)
            decoded_audio, sr = audio_tokenizer.decode(
                revert_delay_pattern(audio_data))
            decoded_audios.append(decoded_audio)
        # asr_text = _get_asr(decoded_audio, sr, asr_pipeline)
        decoded_audio = np.concatenate(decoded_audios)
        sf.write(f"audio_dumps/audio_out_{i}.wav", decoded_audio, sr)
        print(outputs[i].outputs[0].text)
