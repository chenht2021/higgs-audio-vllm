import base64
import os

from vllm import LLM, SamplingParams

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
    "{% endif %}"
)

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
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n<|audio_out_bos|>' }}"
    "{% endif %}"
)

TEST_MODEL_PATH = "/fsx/models/higgs_audio_test_models"


def encode_base64_content_from_file(file_path: str) -> str:
    """Encode a content from a local file to base64 format using librosa."""
    # Read the MP3 file as binary and encode it directly to Base64
    with open(file_path, "rb") as audio_file:
        audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")
    return audio_base64


def test_text_in_audio_out():
    model_path = os.path.join(TEST_MODEL_PATH, "higgs_audio_out_3b_20241222")
    llm = LLM(model=model_path)
    sampling_params = SamplingParams(temperature=0, stop=["<|eot_id|>", "<|end_of_text|>"])
    conversation = [
        {
            "role": "system",
            "content": "You need to generate audio that matches the given speaker description. "
            "You are a young girl. You should sound excited and speak in normal speed. "
            "The user will now give you text, convert the following user texts to speech.",
        },
        {
            "role": "user",
            "content": "Mr. Bounce was very small and like a rubber ball.",
        },
    ]
    outputs = llm.chat(
        conversation,
        sampling_params=sampling_params,
        use_tqdm=False,
        chat_template=AUDIO_OUT_CHAT_TEMPLATE,
    )
    print(outputs)


def test_audio_in_text_out():
    model_path = os.path.join(TEST_MODEL_PATH, "higgs_audio_in_3b_20241222")
    llm = LLM(model=model_path)
    audio_path = "./audiobook_sample.mp3"
    audio_base64 = encode_base64_content_from_file(audio_path)
    file_suffix = audio_path.split(".")[-1]
    conversation = [
        {
            "role": "system",
            "content": "Transcribe the audio.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {
                        # Any format supported by librosa is supported
                        "url": f"data:audio/{file_suffix};base64,{audio_base64}"
                    },
                },
            ],
        },
    ]
    sampling_params = SamplingParams(
        temperature=0, stop=["<|eot_id|>", "<|end_of_text|>"], max_tokens=512
    )
    outputs = llm.chat(
        conversation,
        sampling_params=sampling_params,
        use_tqdm=False,
        chat_template=TEXT_OUT_CHAT_TEMPLATE,
    )
    reference_output = (
        "The spookie just tells you how you're doing up. He is a great ex-pawer. "
        "No grass left, no cajun. There's lots of grass left, there'll be lots of "
        "grass left and just a few some for the rabbits. What rabbit?"
    )
    assert outputs[0].outputs[0].text == reference_output
