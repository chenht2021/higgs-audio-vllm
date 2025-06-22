# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass


@dataclass
class MultimodalMetadata:
    # [batch_size]
    num_audio_eos: list[int]
    # [batch_size]
    num_audio_delays: list[int]
    # [batch_size]
    last_prompt_token_ids: list[int]
    # [batch_size]
    audio_generation_mode: list[int]
