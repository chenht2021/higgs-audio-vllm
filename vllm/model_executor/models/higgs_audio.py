"""Inference-only Higgs Audio model compatible with HuggingFace weights."""
import math
import os
from functools import lru_cache
from typing import (Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple,
                    TypedDict, Union)

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, BatchFeature, ProcessorMixin
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.whisper import WhisperFeatureExtractor
from transformers.models.whisper.modeling_whisper import WhisperEncoderLayer
from transformers.tokenization_utils_base import (PaddingStrategy,
                                                  PreTokenizedInput, TextInput)

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group
from vllm.inputs import InputContext
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    get_compressed_tensors_cache_scale)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.llama import (LlamaAttention,
                                              LlamaDecoderLayer, LlamaMLP)
from vllm.model_executor.models.utils import (extract_layer_index,
                                              is_pp_missing_parameter,
                                              make_layers,
                                              merge_multimodal_embeddings)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import NestedTensors
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        MultiModalDataItems, ProcessorInputs,
                                        PromptReplacement)
from vllm.sequence import IntermediateTensors

from .higgs_audio_config import HiggsAudioConfig, HiggsAudioEncoderConfig

logger = init_logger(__name__)

_KEYS_TO_MODIFY_MAPPING = {
    "audio_decoder_proj.audio_lm_head": "audio_lm_head",
    "audio_decoder_proj.text_lm_head": "text_lm_head",
}

AutoConfig.register("higgs_audio_encoder", HiggsAudioEncoderConfig)
AutoConfig.register("higgs_audio", HiggsAudioConfig)

# # === Audio Inputs === #
class HiggsAudioInputs(TypedDict):
    # (num_audios, num_mel_bins, 3000)`
    audio_features: torch.Tensor

    # (num_audios, 3000)
    audio_feature_attention_mask: torch.Tensor


# Revised on top of transformers.models.qwen2_audio.modeling_qwen2_audio with Qwen2AudioEncoder --> HiggsAudioEncoder
# The code was originally borrowed from WhisperEncoder
class HiggsAudioEncoder(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`WhisperEncoderLayer`].

    Args:
        config: HiggsAudioEncoderConfig
    """

    # Ignore copy
    config_class = HiggsAudioEncoderConfig
    main_input_name = "input_features"
    _no_split_modules = ["WhisperEncoderLayer"]

    def __init__(self, config: HiggsAudioEncoderConfig):
        super().__init__()
        self.config = config
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)

        self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)
        self.embed_positions.requires_grad_(False)

        self.layers = nn.ModuleList(
            [WhisperEncoderLayer(config) for _ in range(config.encoder_layers)]
        )
        self.layer_norm = nn.LayerNorm(config.d_model)
        # Ignore copy
        self.avg_pooler = nn.AvgPool1d(2, stride=2)

        self.gradient_checkpointing = False

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def get_input_embeddings(self) -> nn.Module:
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value

    def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        check_seq_length=True,
    ):
        r"""
        Args:
            input_features (`torch.LongTensor` of shape `(batch_size, feature_size, sequence_length)`):
                Float values of mel features extracted from the raw speech waveform. Raw speech waveform can be
                obtained by loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a
                `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`). To prepare the array into
                `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the mel features, padding
                and conversion into a tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]
            attention_mask (`torch.Tensor`)`, *optional*):
                HiggsAudio does not support masking of the `input_features`, this argument is preserved for compatibility,
                but it is not used. By default the silence in the input log mel spectrogram are ignored.
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """

        expected_seq_length = (
            self.config.max_source_positions * self.conv1.stride[0] * self.conv2.stride[0]
        )
        if check_seq_length and (input_features.shape[-1] != expected_seq_length):
            raise ValueError(
                f"HiggsAudio expects the mel input features to be of length {expected_seq_length}, but found {input_features.shape[-1]}. Make sure to pad the input mel features to {expected_seq_length}."
            )

        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Ignore copy
        input_features = input_features.to(
            dtype=self.conv1.weight.dtype, device=self.conv1.weight.device
        )

        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos = self.embed_positions.weight

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            # Ignore copy
            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__,
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                        output_attentions,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Ignore copy
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.avg_pooler(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1)

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, encoder_states, all_attentions] if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )

    # Ignore copy
    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder
        """
        # TODO(sxjscience) Double confirm the formula
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1
        return input_lengths, output_lengths


class HiggsAudioFeatureProjector(nn.Module):
    """Projector that maps audio features extracted by Whisper to hidden state of the text model."""

    def __init__(self, vllm_config: VllmConfig):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.linear = nn.Linear(
            config.audio_encoder_config.d_model, config.text_config.hidden_size, bias=True
        )

    def forward(self, audio_features):
        hidden_states = self.linear(audio_features)
        return hidden_states


class HiggsAudioDecoderProjector(nn.Module):
    """Projection layers that map hidden states from the LLM component to audio / text logits."""

    def __init__(self, vllm_config: VllmConfig):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self._audio_decoder_proj_num_layers = config.audio_decoder_proj_num_layers
        if self._audio_decoder_proj_num_layers > 0:
            self.transformer_layers = nn.ModuleList(
                [
                    LlamaDecoderLayer(
                        config.text_config, layer_idx + config.text_config.num_hidden_layers
                    )
                    for layer_idx in range(config.audio_decoder_proj_num_layers)
                ]
            )

            is_neox_style = True
            if quant_config is not None and quant_config.get_name() == "gguf":
                is_neox_style = False
            self.rotary_emb = get_rope(
                head_dim=config.text_config.head_dim,
                rotary_dim=config.text_config.head_dim,
                max_position=config.text_config.max_position_embeddings,
                base=config.text_config.rope_theta,
                rope_scaling=config.text_config.repe_scaling,
                is_neox_style=is_neox_style,
            )
            self.norm = RMSNorm(config.text_config.hidden_size, eps=config.text_config.rms_norm_eps)

    def forward(
        self,
        hidden_states,
        audio_out_mask=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        cache_position=None,
    ):
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, seq_len, hidden_size)`):
                Hidden states from the LLM component
            audio_out_mask (`torch.Tensor` of shape `(batch_size, seq_len)`):
                Mask for identifying the audio out tokens.
            attention_mask (`torch.Tensor` of shape `(batch_size, seq_len)`):
                Mask to avoid performing attention on padding token indices
            position_ids (`torch.Tensor` of shape `(batch_size, seq_len)`):
                Position ids for the input tokens

        Returns:
            logits (`torch.Tensor` of shape `(batch_size, seq_len, vocab_size)`):
                Logits for text tokens
            audio_logits (`torch.Tensor` of shape `(num_audio_out_tokens, audio_num_codebooks * audio_codebook_size)`):
                Logits for audio tokens. We ensure `num_text_tokens + num_audio_tokens == batch_size * seq_len`. If we
                the model only outputs text logits, `audio_logits` will be `None`.

        """
        # TODO(sxjscience) Need to check if DeepSpeed Zero3 supports zero-shape input.
        if self._audio_decoder_proj_num_layers > 0:
            # create position embeddings to be shared across the decoder layers
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
            for decoder_layer in self.transformer_layers:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
                hidden_states = layer_outputs[0]
            hidden_states = self.norm(hidden_states)

        return hidden_states


def get_processor(
    tokenzier,
    *args,
    trust_remote_code: bool = False,
    **kwargs,
):
    """Gets a processor for the given model name via HuggingFace.

    Derived from `vllm.transformers_utils.image_processor.get_image_processor`.
    """
    # don't put this import at the top level
    # it will call torch.cuda.device_count()
    from transformers import AutoFeatureExtractor

    try:
        feature_extractor_name = os.getenv("HIGGS_AUDIO_TOKENIZER", "openai/whisper-large-v3-turbo")
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            feature_extractor_name,  # TODO: Write into config file
            *args,
            trust_remote_code=trust_remote_code,
            attn_implementation="sdpa",
            **kwargs,
        )
        processor = HFHiggsAudioProcessor(
            feature_extractor=feature_extractor,
            tokenizer=tokenzier,
        )
        logger.info("Loaded HFHiggsAudioProcessor")
    except ValueError as e:
        # If the error pertains to the processor class not existing or not
        # currently being imported, suggest using the --trust-remote-code flag.
        # Unlike AutoTokenizer, AutoProcessor does not separate such errors
        if not trust_remote_code:
            err_msg = (
                "Failed to load the processor. If the processor is "
                "a custom processor not yet available in the HuggingFace "
                "transformers library, consider setting "
                "`trust_remote_code=True` in LLM or using the "
                "`--trust-remote-code` flag in the CLI."
            )
            raise RuntimeError(err_msg) from e
        else:
            raise e

    return processor


cached_get_processor = lru_cache(get_processor)


def _get_feat_extract_output_lengths(input_lengths: torch.LongTensor):
    """
    Computes the output length of the convolutional layers
    and the output length of the audio encoder
    """
    input_lengths = (input_lengths - 1) // 2 + 1
    output_lengths = (input_lengths - 2) // 2 + 1
    return input_lengths, output_lengths


def get_max_higgs_audio_audio_tokens(ctx: InputContext) -> int:
    max_source_position = ctx.model_config.hf_config.audio_encoder_config.max_source_positions
    output_lengths = (max_source_position - 2) // 2 + 1
    return output_lengths


class HFHiggsAudioProcessor(ProcessorMixin):
    """
    HF Processor class for Higgs audio model. Mostly borrow from processing_qwen2_audio.py.
    """

    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "WhisperFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, feature_extractor=None, tokenizer=None, chat_template=None):
        if chat_template is None:
            chat_template = self.default_chat_template
        super().__init__(feature_extractor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        audios: Union[np.ndarray, List[np.ndarray]] = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        sampling_rate: Optional[int] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and audio(s). Borrowed the code from
        Qwen2 Audio.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            audios (`np.ndarray`, `List[np.ndarray]`):
                The audio or batch of audios to be prepared. Each audio can be a NumPy array.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:
                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            sampling_rate (`int`, defaults to 16000):
                The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        """

        if text is None:
            raise ValueError("You need to specify either a `text` input to process.")
        inputs = self.tokenizer(text, padding=padding, **kwargs)

        if audios is not None:
            audio_inputs = self.feature_extractor(
                audios,
                sampling_rate=sampling_rate,
                return_attention_mask=True,
                padding="max_length",
                **kwargs,
            )
            # Rename to audio_feature_attention_mask to prevent conflicts
            # with text attention mask
            audio_inputs["audio_feature_attention_mask"] = audio_inputs.pop("attention_mask")
            inputs.update(audio_inputs)

        return BatchFeature(data={**inputs})

    @property
    def default_chat_template(self):
        # fmt: off
        return (
            "{% set loop_messages = messages %}"
            "{% for message in loop_messages %}"
                "<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n"
                "{% if message['content'] is string %}"
                    "{% set content = message['content'] | trim + '<|eot_id|>' %}"
                "{% else %}"
                    "{% for content in message['content'] %}"
                        "{% if 'audio' in content or 'audio_url' in content %}"
                            "{% set content = '<|audio_bos|><|AUDIO|><|audio_eos|>' %}"
                        "{% elif 'text' in content %}"
                            "{% set content = content['text'] %}"
                        "{% endif %}"
                    "{% endfor %}"
                "{% endif %}"
                "{% if loop.index0 == 0 %}"
                    "{% set content = bos_token + content %}"
                "{% endif %}"
                "{{ content }}<|eot_id|>"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
            "{% endif %}"
        )
        # fmt: on


class HiggsAudioMultiModalProcessor(BaseMultiModalProcessor):
    def _get_hf_processor(self):
        return cached_get_processor(self.ctx.tokenizer)

    def _get_feature_extractor(self) -> WhisperFeatureExtractor:
        return self._get_hf_processor().feature_extractor  # type: ignore

    def _get_processor_data(
        self,
        mm_items: MultiModalDataItems,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        # resample audio to the model's sampling rate
        feature_extractor = self._get_feature_extractor()
        mm_items.resample_audios(feature_extractor.sampling_rate)

        return super()._get_processor_data(mm_items)

    def _call_hf_processor(
        self,
        hf_processor: ProcessorMixin,
        prompt: str,
        processor_data: Mapping[str, object],
        mm_processor_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        processor_data = dict(processor_data)
        audios = processor_data.pop("audios", [])

        if audios:
            processor_data["audios"] = audios

            feature_extractor = self._get_feature_extractor()
            mm_processor_kwargs = dict(
                **mm_processor_kwargs,
                sampling_rate=feature_extractor.sampling_rate,
            )
        else:
            # NOTE: WhisperFeatureExtractor cannot handle empty list of audios
            pass

        batch_data = super()._call_hf_processor(
            hf_processor,
            prompt=prompt,
            processor_data=processor_data,
            mm_processor_kwargs=mm_processor_kwargs,
        )
        batch_data["audio_features"] = batch_data.pop("input_features")
        return batch_data

    def _get_prompt_replacements(
        self,
        mm_items: MultiModalDataItems,
        hf_inputs: BatchFeature,
        mm_processor_kwargs: Mapping[str, object],
    ) -> list[PromptReplacement]:
        hf_config = self.ctx.get_hf_config(HiggsAudioConfig)
        placeholder = hf_config.audio_in_token_idx

        audio_feature_attention_mask = hf_inputs.get("audio_feature_attention_mask")
        if audio_feature_attention_mask is None:
            audio_output_lengths = []
        else:
            _, audio_output_lengths = _get_feat_extract_output_lengths(
                audio_feature_attention_mask.sum(-1)
            )

        def get_replacement_higgs_audio(item_idx: int):
            return [placeholder] * audio_output_lengths[item_idx]

        return [
            PromptReplacement(
                modality="audio",
                target=[placeholder],
                replacement=get_replacement_higgs_audio,
            )
        ]

    def _get_dummy_mm_inputs(
        self,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
        audio_len = get_max_higgs_audio_audio_tokens(self.ctx)

        audio_count = mm_counts["audio"]
        audio = np.zeros(audio_len)
        data = {"audio": [audio] * audio_count}

        return ProcessorInputs(
            prompt_text="<|AUDIO|>" * audio_count,
            mm_data=data,
            mm_processor_kwargs={},
        )


class HiggsAudioDualFFNDecoderLayer(nn.Module):
    """We implement a dual-path FFN decoder layer where the audio tokens and text tokens go through separate FFN layers.

    The audio and text tokens share the text-attention layer, but will be encoded with separate feedforward layers.
    In addition, the audio tokens can be configured to go through separate attention layer.

    Following is an illustration:

     t    t    t    a   a     a    t    t    t
                        |
                        | (audio self-attention layer)
                        v
    t    t     t    h'_a h'_a  h'_a  t  t    t
                        |
                        | (shared attention layer)
                        v
    h_t  h_t  h_t  h_a  h_a  h_a  h_t  h_t  h_t
                        |
                        | (separate text/audio hidden states)
                        v
    [h_t  h_t  h_t  h_t  h_t  h_t], [h_a, h_a, h_a]
             |                             |
             | (separate FFNs)             |
             v                             v
    [o_t  o_t  o_t  o_t  o_t  o_t], [o_a, o_a, o_a]
                        |
                        | (reorder)
                        v
    o_t  o_t  o_t  o_a  o_a  o_a  o_t  o_t  o_t

    This has a few advantages:
    1) We are able to use a smaller FFN, or even bypass the FFN for audio tokens. This accelerates the inference speed.
    2) The Audio-FFN introduces more trainable parameters to the model.
       This should have the same effect as the mixture-of-expert layer and we may expect better performance due to the scaling law.
    3) We can replace the original FFN in LLMs with the dual-path FFN without changing the model architecture.


    """

    def __init__(
        self,
        config: HiggsAudioConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        fast_forward: bool = False,
        use_audio_attention: bool = False,
    ):
        super().__init__()
        text_config = config.text_config
        self.hidden_size = text_config.hidden_size
        self.layer_idx = extract_layer_index(prefix)
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(config, "original_max_position_embeddings", None):
            rope_scaling[
                "original_max_position_embeddings"
            ] = config.original_max_position_embeddings
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        attention_bias = getattr(config, "attention_bias", False) or getattr(config, "bias", False)
        self.self_attn = LlamaAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads", config.num_attention_heads),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = LlamaMLP(text_config)

        if not fast_forward:
            if use_audio_attention:
                self.audio_attn = LlamaAttention(
                    config=config,
                    hidden_size=self.hidden_size,
                    num_heads=config.num_attention_heads,
                    num_kv_heads=getattr(config, "num_key_value_heads", config.num_attention_heads),
                    rope_theta=rope_theta,
                    rope_scaling=rope_scaling,
                    max_position_embeddings=max_position_embeddings,
                    quant_config=quant_config,
                    bias=attention_bias,
                    cache_config=cache_config,
                    prefix=f"{prefix}.self_attn",
                )
                self.audio_post_audio_attn_layer_norm = RMSNorm(
                    text_config.hidden_size, eps=text_config.rms_norm_eps
                )

            self.audio_mlp = LlamaMLP(text_config)
            self.audio_input_layernorm = RMSNorm(
                text_config.hidden_size, eps=text_config.rms_norm_eps
            )
            self.audio_post_attention_layernorm = RMSNorm(
                text_config.hidden_size, eps=text_config.rms_norm_eps
            )

        self.use_audio_attention = use_audio_attention
        self.fast_forward = fast_forward
        if self.fast_forward:
            assert (
                not self.use_audio_attention
            ), "We cannot use audio_attention if the layer is marked as fast-forward."
        self.input_layernorm = RMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            text_config.hidden_size, eps=text_config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        audio_out_mask: Optional[torch.BoolTensor],
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            position_ids
                IDs of positions in the input sequence
            audio_out_mask
                Mask for identifying the audio tokens. Size (batch_size, sequence_length)
                1 --> location contains audio_out
                0 --> location does not contain audio_out

                When use_cache is True, the audio_out_mask contains audio_out masks for all tokens up to the current token.
                That means, it has size (batch_size, sequence_length) while hidden_states will have size (batch_size, 1)
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        assert (
            residual is None
        ), "The residual output is not supported in HiggsAudioDualFFNDecoderLayer."
        residual = hidden_states
        has_audio_out = audio_out_mask is not None and audio_out_mask.shape[0] > 0

        if self.fast_forward and has_audio_out:
            original_hidden_states = hidden_states.clone()

        if not self.fast_forward and has_audio_out:
            hidden_states = torch.where(
                audio_out_mask.unsqueeze(-1),
                self.audio_input_layernorm(hidden_states),
                self.input_layernorm(hidden_states),
            )
        else:
            hidden_states = self.input_layernorm(hidden_states)

        # Audio Attention
        if self.use_audio_attention and has_audio_out:
            assert (
                kv_cache.shape[0] == 4
            ), "The KV cache should have shape (4, batch_size, seq_len, hidden_size)"
            audio_hidden_states = self.audio_attn(
                positions=positions,
                hidden_states=hidden_states,
                kv_cache=kv_cache[2:4],
                attn_metadata=attn_metadata,
            )
            audio_hidden_states = residual + audio_hidden_states
            residual = torch.where(audio_out_mask.unsqueeze(-1), audio_hidden_states, residual)
            audio_hidden_states = self.audio_post_audio_attn_layer_norm(audio_hidden_states)
            hidden_states = torch.where(
                audio_out_mask.unsqueeze(-1), audio_hidden_states, hidden_states
            )

        # Text Attention
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache[0:2],
            attn_metadata=attn_metadata,
        )
        hidden_states = residual + hidden_states

        # Apply Dual-path FFN
        residual = hidden_states

        if has_audio_out and not self.fast_forward:
            text_hidden_states = self.post_attention_layernorm(hidden_states[~audio_out_mask])
            audio_hidden_states = self.audio_post_attention_layernorm(hidden_states[audio_out_mask])

            text_hidden_states = self.mlp(text_hidden_states)
            residual[~audio_out_mask] += text_hidden_states

            audio_hidden_states = self.audio_mlp(audio_hidden_states)
            residual[audio_out_mask] += audio_hidden_states

            hidden_states = residual
        else:
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states

        if self.fast_forward and has_audio_out:
            hidden_states = torch.where(
                audio_out_mask.unsqueeze(-1), original_hidden_states, hidden_states
            )

        # Add a None as the residual output for the compatibility
        outputs = (hidden_states, None)

        return outputs


@MULTIMODAL_REGISTRY.register_max_multimodal_tokens("audio", get_max_higgs_audio_audio_tokens)
@MULTIMODAL_REGISTRY.register_processor(HiggsAudioMultiModalProcessor)
class HiggsAudioForConditionalGeneration(nn.Module, SupportsMultiModal):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config

        self.multimodal_config = multimodal_config

        # Force to set attention implementation
        config.audio_encoder_config._attn_implementation = "sdpa"
        self.audio_tower = HiggsAudioEncoder(config.audio_encoder_config)

        self.quant_config = quant_config

        self.embed_tokens = nn.Embedding(
            config.text_config.vocab_size, config.text_config.hidden_size, config.pad_token_id
        )

        if config.audio_adapter_type == "dual_ffn_fast_forward":
            self.start_layer, self.end_layer, self.layers = make_layers(
                config.text_config.num_hidden_layers,
                lambda prefix: HiggsAudioDualFFNDecoderLayer(
                    config=config.text_config,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers",
                ),
                prefix=f"{prefix}.layers",
            )
        elif config.audio_adapter_type == "stack":
            self.start_layer, self.end_layer, self.layers = make_layers(
                config.text_config.num_hidden_layers,
                lambda prefix: LlamaDecoderLayer(
                    config=config.text_config,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers",
                ),
                prefix=f"{prefix}.layers",
            )
        else:
            raise NotImplementedError(
                f"Audio adapter type {config.audio_adapter_type} not implemented."
            )
        self.norm = RMSNorm(config.text_config.hidden_size, eps=config.text_config.rms_norm_eps)

        is_neox_style = True
        if quant_config is not None and quant_config.get_name() == "gguf":
            is_neox_style = False
        self.rotary_emb = get_rope(
            head_size=config.text_config.head_dim,
            rotary_dim=config.text_config.head_dim,
            max_position=config.text_config.max_position_embeddings,
            base=config.text_config.rope_theta,
            rope_scaling=config.text_config.rope_scaling,
            is_neox_style=is_neox_style,
        )

        self.audio_encoder_proj = HiggsAudioFeatureProjector(vllm_config)
        self.audio_codebook_size = (
            config.audio_codebook_size + 2
        )  # We add 1 for the audio_stream_bos token and 1 for the audio_stream_eos token
        self.use_audio_out = config.audio_decoder_proj_num_layers

        if config.use_audio_out_embed_projector:
            self.audio_out_embed_projector = nn.Linear(
                config.text_config.hidden_size, config.text_config.hidden_size, bias=False
            )

        self.audio_codebook_embeddings = nn.Embedding(
            config.audio_num_codebooks * self.audio_codebook_size, config.text_config.hidden_size
        )

        self.text_lm_head = ParallelLMHead(
            config.text_config.vocab_size,
            config.text_config.hidden_size,
            quant_config=quant_config,
            bias=False,
        )

        if self.use_audio_out:
            self.audio_lm_head = ParallelLMHead(
                config.text_config.hidden_size,
                config.audio_num_codebooks * (config.audio_codebook_size + 2),
                quant_config=quant_config,
                bias=False,
            )

        if get_pp_group().is_last_rank:
            self.audio_decoder_proj = HiggsAudioDecoderProjector(vllm_config)
            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(
                config.text_config.vocab_size, config.text_config.vocab_size, logit_scale
            )
            self.sampler = get_sampler()

    def _validate_and_reshape_mm_tensor(
        self,
        mm_input: object,
        name: str,
        pad_with: Optional[int] = None,
    ) -> torch.Tensor:
        if not isinstance(mm_input, (torch.Tensor, list)):
            raise ValueError(f"Incorrect type of {name}. " f"Got type: {type(mm_input)}")
        if isinstance(mm_input, torch.Tensor):
            return torch.concat(list(mm_input))
        else:
            if pad_with is not None:
                max_size = max(
                    [tensor.size(-1) for tensor in mm_input]
                )  # Find max size along the last dimension
                # Step 2: Pad each tensor to the max size along the last dimension
                padded_tensors = []
                for tensor in mm_input:
                    pad_size = max_size - tensor.size(-1)  # Calculate how much padding is needed
                    if pad_size > 0:
                        # Pad tensor along the last dimension (right side)
                        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_size))
                    else:
                        padded_tensor = tensor
                    padded_tensors.append(padded_tensor)
                return torch.concat(padded_tensors)
            else:
                return torch.concat(mm_input)

    def _parse_and_validate_audio_input(self, **kwargs: object) -> Optional[HiggsAudioInputs]:
        audio_features = kwargs.pop("audio_features", None)
        audio_feature_attention_mask = kwargs.pop("audio_feature_attention_mask", None)
        if audio_features is None:
            return None
        audio_features = self._validate_and_reshape_mm_tensor(audio_features, "audio_features")
        audio_feature_attention_mask = self._validate_and_reshape_mm_tensor(
            audio_feature_attention_mask,
            "audio_feature_attention_mask",
            pad_with=0,
        )
        if not isinstance(audio_features, (torch.Tensor, list)):
            raise ValueError(
                "Incorrect type of audio input features. " f"Got type: {type(audio_features)}"
            )
        return HiggsAudioInputs(
            audio_features=audio_features, audio_feature_attention_mask=audio_feature_attention_mask
        )

    def _process_audio_input(self, audio_input: HiggsAudioInputs) -> torch.Tensor:
        audio_features = audio_input["audio_features"]
        audio_feature_attention_mask = audio_input["audio_feature_attention_mask"]

        (
            audio_feat_lengths,
            audio_feat_out_lengths,
        ) = self.audio_tower._get_feat_extract_output_lengths(audio_feature_attention_mask.sum(-1))

        batch_size, _, max_mel_seq_len = audio_features.shape
        max_seq_len = (max_mel_seq_len - 2) // 2 + 1
        # Create a sequence tensor of shape (batch_size, max_seq_len)
        seq_range = (
            torch.arange(
                0, max_seq_len, dtype=audio_feat_lengths.dtype, device=audio_feat_lengths.device
            )
            .unsqueeze(0)
            .expand(batch_size, max_seq_len)
        )
        lengths_expand = audio_feat_lengths.unsqueeze(-1).expand(batch_size, max_seq_len)
        # Create mask
        padding_mask = seq_range >= lengths_expand

        audio_attention_mask_ = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(
            batch_size, 1, max_seq_len, max_seq_len
        )
        audio_attention_mask = audio_attention_mask_.to(
            dtype=self.audio_tower.conv1.weight.dtype, device=self.audio_tower.conv1.weight.device
        )
        audio_attention_mask[audio_attention_mask_] = float("-inf")

        audio_outputs = self.audio_tower(audio_features, attention_mask=audio_attention_mask)
        selected_audio_feature = audio_outputs.last_hidden_state
        audio_features = self.audio_encoder_proj(selected_audio_feature)

        num_audios, max_audio_tokens, embed_dim = audio_features.shape
        audio_features_mask = torch.arange(max_audio_tokens).expand(
            num_audios, max_audio_tokens
        ).to(audio_feat_out_lengths.device) < audio_feat_out_lengths.unsqueeze(1)
        masked_audio_features = audio_features[audio_features_mask].view(-1, embed_dim)

        return masked_audio_features

    def get_multimodal_embeddings(self, **kwargs) -> Optional[NestedTensors]:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is None:
            return None
        masked_audio_features = self._process_audio_input(audio_input)
        return masked_audio_features

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[NestedTensors] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.embed_tokens(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings, self.config.audio_in_token_idx
            )
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            # NOTE: In v1, inputs_embeds is always generated at model runner, this
            # condition is for v0 compatibility.
            if inputs_embeds is None:
                multimodal_embeddings = self.get_multimodal_embeddings(**kwargs)
                inputs_embeds = self.get_input_embeddings(input_ids, multimodal_embeddings)
                input_ids = None
            hidden_states = inputs_embeds
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            if isinstance(layer, HiggsAudioDualFFNDecoderLayer):
                hidden_states, _ = layer(
                    positions=positions,
                    hidden_states=hidden_states,
                    audio_out_mask=None,  # FIXME
                    kv_cache=kv_caches[i - self.start_layer],
                    attn_metadata=attn_metadata,
                    residual=None,
                )
            else:
                hidden_states, residual = layer(
                    positions,
                    hidden_states,
                    kv_caches[i - self.start_layer],
                    attn_metadata,
                    residual,
                )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})
        hidden_states, _ = self.norm(hidden_states, residual)
        if self.use_audio_out:
            hidden_states = self.audio_decoder_proj(hidden_states)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        lm_head = self.audio_lm_head if self.use_audio_out else self.text_lm_head
        logits = self.logits_processor(lm_head, hidden_states, sampling_metadata)
        return logits

    def sample(
        self, logits: torch.Tensor, sampling_metadata: SamplingMetadata
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            # Skip audio_lm_head if we are not using audio out.
            if not self.use_audio_out and "audio_lm_head" in name:
                continue
            for key_to_modify, new_key in _KEYS_TO_MODIFY_MAPPING.items():
                if key_to_modify in name:
                    name = name.replace(key_to_modify, new_key)

            if scale_name := get_compressed_tensors_cache_scale(name):
                # Loading kv cache scales for compressed-tensors quantization
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = loaded_weight[0]
                weight_loader(param, loaded_weight)
                continue

            if "audio_tower" in name:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)

                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
