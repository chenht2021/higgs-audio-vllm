# SPDX-License-Identifier: Apache-2.0
import base64
from functools import lru_cache
from typing import Optional

import numpy as np

from vllm.entrypoints.openai.protocol import ChatCompletionAudio
from vllm.logger import init_logger
from vllm.model_executor.models.higgs_audio_tokenizer import (
    AudioTokenizer, revert_delay_pattern)
from vllm.utils import random_uuid

logger = init_logger(__name__)


def token2wav(
    token: np.ndarray,
    audio_chunk_size: int,
    is_first_chunk: bool,
    audio_tokenizer: AudioTokenizer,
    audio_codebook_size: int,
    samples_per_token: int,
    audio_num_codebooks: int,
    fade_out_audio: Optional[np.ndarray] = None,
    finalize: bool = False,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    # The input token is (# of tokens, # of codebooks)
    # Transpose to (# of codebooks, # of tokens)
    token = token.transpose(1, 0)

    if token.shape[1] <= audio_num_codebooks + 2:
        logger.warning(
            "The audio token length %s is too short. Skipping this chunk.",
            token.shape[1])
        return None, None

    audio_codes = revert_delay_pattern(token) \
                    .clip(0, audio_codebook_size - 1)
    # Remove the very first audio bos token from the input token
    if is_first_chunk:
        audio_codes = audio_codes[:, 1:]
        audio_chunk_size -= 1

    tts_speech, _ = audio_tokenizer.decode(vq_code=audio_codes)
    if fade_out_audio is not None:
        hamming_window_len = min(2 * len(fade_out_audio), samples_per_token)
        hamming_window = _get_hamming_window(hamming_window_len)
        fade_overlap = hamming_window_len // 2
        tts_speech[:fade_overlap] = \
            tts_speech[:fade_overlap] * hamming_window[:fade_overlap] + \
            fade_out_audio[:fade_overlap] * hamming_window[fade_overlap:]

    fade_out_audio = tts_speech[audio_chunk_size * samples_per_token:]
    if not finalize:
        tts_speech = tts_speech[:audio_chunk_size * samples_per_token]
    else:
        fade_out_audio = None
    return tts_speech, fade_out_audio


def create_audio_chunk(
    audio_tokens_cache: np.ndarray,
    audio_chunk_size: int,
    fade_out_audio: Optional[np.ndarray],
    is_first_chunk: bool,
    audio_tokenizer: AudioTokenizer,
    audio_codebook_size: int,
    samples_per_token: int,
    audio_num_codebooks: int,
    finalize: bool = False,
    return_as_numpy_audio: bool = False
) -> tuple[Optional[ChatCompletionAudio], np.ndarray]:
    new_audio, new_fade_out_audio = token2wav(
        audio_tokens_cache,
        audio_chunk_size,
        is_first_chunk=is_first_chunk,
        fade_out_audio=fade_out_audio,
        finalize=finalize,
        audio_tokenizer=audio_tokenizer,
        audio_codebook_size=audio_codebook_size,
        samples_per_token=samples_per_token,
        audio_num_codebooks=audio_num_codebooks,
    )

    if return_as_numpy_audio:
        return new_audio, new_fade_out_audio

    audio_pcm16 = (new_audio * np.iinfo(np.int16).max).astype(np.int16)

    return ChatCompletionAudio(
        id=f"audio-{random_uuid()}",
        data=base64.b64encode(audio_pcm16).decode("utf-8"),
        expires_at=0,
        transcript="",
    ), new_fade_out_audio


@lru_cache
def _get_hamming_window(len):
    return np.hamming(len)
