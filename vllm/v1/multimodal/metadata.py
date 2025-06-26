# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class MultimodalMetadata:
    # [batch_size]
    num_audio_eos: list[int]
    # [batch_size]
    num_audio_delays: list[int]
    # [batch_size]
    last_prompt_token_ids: list[int]
    # [num_tokens, ]: The modality of each token.
    token_mm_map: Optional[torch.Tensor] = None
