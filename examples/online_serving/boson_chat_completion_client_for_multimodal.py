# SPDX-License-Identifier: Apache-2.0
"""An example showing how to use vLLM to serve multimodal models 
and run online inference with OpenAI client.
"""
import argparse
import base64
import os
from io import BytesIO

import numpy as np
import requests
import soundfile as sf
from openai import OpenAI


def encode_base64_content_from_file(file_path: str) -> str:
    """Encode a content from a local file to base64 format."""
    # Read the MP3 file as binary and encode it directly to Base64
    with open(file_path, "rb") as audio_file:
        audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")
    return audio_base64


def encode_base64_content_from_url(content_url: str) -> str:
    """Encode a content retrieved from a remote url to base64 format."""

    with requests.get(content_url) as response:
        response.raise_for_status()
        result = base64.b64encode(response.content).decode("utf-8")

    return result


def run_text_only() -> None:
    chat_completion = client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": "What's the capital of France?"
        }],
        model=model,
        max_completion_tokens=64,
    )

    result = chat_completion.choices[0].message.content
    print("Chat completion output:", result)


def run_audio_generation() -> None:
    chat_completion = client.chat.completions.create(
        messages=[{
            "role":
            "system",
            "content":
            ("You need to generate audio that matches the given "
             "speaker description. You are a young girl. You should "
             "sound excited and speak in normal speed. The user will "
             "now give you text, convert the following user texts "
             "to speech.")
        }, {
            "role":
            "user",
            "content":
            ("Mr. Bounce was very small and like a rubber ball. He "
             "loved bouncing around the neighborhood, bringing joy "
             "to all the children who saw him. Up and down he would "
             "go, higher and higher each time, his cheerful laughter "
             "echoing through the streets. The local kids would "
             "gather to watch his amazing acrobatics as he bounced "
             "over fences, onto rooftops, and even through open "
             "windows. Mr. Bounce's greatest dream was to bounce "
             "all the way to the moon one day, and the way he kept "
             "reaching new heights, the children believed he just "
             "might make it.")
        }],
        model=model,
        max_completion_tokens=300,
        modalities=["text", "audio"],
        audio={"format": "mp3"},
    )

    text = chat_completion.choices[0].message.content
    audio = chat_completion.choices[0].message.audio.data
    # Decode base64 audio string to bytes
    audio_bytes = base64.b64decode(audio)
    print("Chat completion text output:", text)
    print("Saving the audio to file")
    with open("output.mp3", "wb") as f:
        f.write(audio_bytes)


def run_tts(stream: bool = False) -> None:
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..",
                            "tests/models/decoder_only/audio_language")
    target_rate = 16000
    audio_path = os.path.join(data_dir, "en_woman_1.wav")
    audio_base64 = encode_base64_content_from_file(audio_path)
    messages = [{
        "role":
        "system",
        "content":
        "Convert the following text from the user into speech."
    }, {
        "role":
        "user",
        "content":
        ("The device would work during the day as well, if you took "
         "steps to either block direct sunlight or point it away from "
         "the sun.")
    }, {
        "role":
        "assistant",
        "content": [{
            "type": "input_audio",
            "input_audio": {
                "data": audio_base64,
                "format": "wav",
            },
        }],
    }, {
        "role":
        "user",
        "content":
        ("In a small town nestled between rolling green hills, there lived "
         "an old man named Elias. He was once a celebrated violinist, but "
         "time had taken his music from him. His hands, once steady, "
         "trembled with age, and his violin sat untouched in its case for "
         "many years.")
    }]
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
        max_completion_tokens=500,
        stream=stream,
        modalities=["text", "audio"],
        temperature=0.7,
    )
    if stream:
        audio_bytes_io = BytesIO()
        i = 0
        for chunk in chat_completion:
            if chunk.choices and hasattr(
                    chunk.choices[0].delta,
                    'audio') and chunk.choices[0].delta.audio:
                audio_bytes = base64.b64decode(
                    chunk.choices[0].delta.audio["data"])
                audio_bytes_io.write(audio_bytes)
                audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                sf.write(f"output_tts_{i}.wav", audio_data, target_rate)
                i += 1
        audio_bytes_io.seek(0)
        audio_data = np.frombuffer(audio_bytes_io.getvalue(), dtype=np.int16)
        print("Saving the audio to file")
        sf.write("output_tts.wav", audio_data, target_rate)
    else:
        text = chat_completion.choices[0].message.content
        audio = chat_completion.choices[0].message.audio.data
        audio_bytes = base64.b64decode(audio)
        print("Chat completion text output:", text)
        print("Saving the audio to file")
        with open("output_tts.wav", "wb") as f:
            f.write(audio_bytes)


def run_generate_dialogue(stream: bool = False) -> None:
    target_rate = 16000
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..",
                            "tests/models/decoder_only/audio_language")
    audio_path = os.path.join(data_dir, "en_woman_1.wav")
    audio_woman_base64 = encode_base64_content_from_file(audio_path)
    audio_path = os.path.join(data_dir, "en_man_1.wav")
    audio_man_base64 = encode_base64_content_from_file(audio_path)

    messages = [{
        "role":
        "user",
        "content":
        ("[SPEAKER0] The device would work during the day as well, "
         "if you took steps to either block direct sunlight or point it "
         "away from the sun.")
    }, {
        "role":
        "assistant",
        "content": [{
            "type": "input_audio",
            "input_audio": {
                "data": audio_woman_base64,
                "format": "wav",
            },
        }],
    }, {
        "role":
        "user",
        "content":
        ("[SPEAKER1] Maintaining your ability to learn translates into "
         "increased marketability, improved career options and higher "
         "salaries.")
    }, {
        "role":
        "assistant",
        "content": [{
            "type": "input_audio",
            "input_audio": {
                "data": audio_man_base64,
                "format": "wav",
            },
        }],
    }, {
        "role":
        "user",
        "content":
        ("[SPEAKER0] Hello, how are you doing today?\n[SPEAKER1] I'm doing "
         "great, thank you!")
    }]
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
        stream=stream,
        stream_options={"include_usage": True},
        stop=["<|end_of_text|>", "<|eot_id|>", "<|audio_eos|>"],
        modalities=["text", "audio"],
        temperature=0.7,
        # extra_body={
        #     "ras_win_len": 7,
        #     "ras_win_max_num_repeat": 2,
        # }
    )

    if stream:
        audio_bytes_io = BytesIO()
        i = 0
        for chunk in chat_completion:
            if chunk.choices and hasattr(
                    chunk.choices[0].delta,
                    'audio') and chunk.choices[0].delta.audio:
                audio_bytes = base64.b64decode(
                    chunk.choices[0].delta.audio["data"])
                audio_bytes_io.write(audio_bytes)
                audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                # sf.write(f"output_tts_{i}.wav", audio_data, target_rate)
                i += 1
            else:
                print(chunk)
        audio_bytes_io.seek(0)
        audio_data = np.frombuffer(audio_bytes_io.getvalue(), dtype=np.int16)
        print("Saving the audio to file")
        sf.write("output_dialogue.wav", audio_data, target_rate)
    else:
        text = chat_completion.choices[0].message.content
        audio = chat_completion.choices[0].message.audio.data
        audio_bytes = base64.b64decode(audio)
        print("Chat completion text output:", text)
        print("Saving the audio to file")
        with open("output_dialogue.wav", "wb") as f:
            f.write(audio_bytes)


def main(args) -> None:
    if args.task == "tts":
        run_tts(args.stream)
    elif args.task == "audio_generation":
        run_audio_generation()
    elif args.task == "text_only":
        run_text_only()
    elif args.task == "dialogue":
        run_generate_dialogue(args.stream)
    else:
        raise ValueError(f"Task {args.task} not supported")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api-base",
        type=str,
        default="http://localhost:8000/v1",
        help="API base URL for OpenAI client.",
    )
    parser.add_argument("--api-key",
                        type=str,
                        default="EMPTY",
                        help="API key for OpenAI client.")
    parser.add_argument("--stream",
                        action="store_true",
                        help="Stream the audio.")
    parser.add_argument(
        "--task",
        type=str,
        default="tts",
        help="Task to run.",
        choices=["tts", "audio_generation", "text_only", "dialogue"])
    args = parser.parse_args()

    client = OpenAI(
        api_key=args.api_key,
        base_url=args.api_base,
    )

    models = client.models.list()
    model = models.data[0].id

    main(args)
