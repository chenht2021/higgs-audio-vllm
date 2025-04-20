# SPDX-License-Identifier: Apache-2.0
from vllm.v1.request import Request, RequestStatus


def check_stop(request: Request, max_model_len: int) -> bool:
    if (request.num_tokens >= max_model_len
            or request.num_output_tokens >= request.max_tokens):
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        return True

    sampling_params = request.sampling_params
    last_token_id = request.output_token_ids[-1]
    if (not sampling_params.ignore_eos
            and last_token_id == request.eos_token_id):
        request.status = RequestStatus.FINISHED_STOPPED
        return True

    if last_token_id in (sampling_params.stop_token_ids or ()):
        request.status = RequestStatus.FINISHED_STOPPED
        request.stop_reason = last_token_id
        return True

    # HACK: Stop for the delay pattern.
    if request._audio_num_codebooks is not None and \
        request._num_audio_eos >= request._audio_num_codebooks:
        request.status = RequestStatus.FINISHED_STOPPED
        request.stop_reason = request.output_mm_token_ids[-1][-1]
        return True

    return False
