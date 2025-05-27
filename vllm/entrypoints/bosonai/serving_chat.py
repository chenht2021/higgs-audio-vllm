# SPDX-License-Identifier: Apache-2.0
import asyncio
import base64
import io
import json
import time
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Callable, Final, Optional, Union

import jinja2
import numpy as np
import soundfile as sf
from fastapi import Request
from pydantic import TypeAdapter

from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import (ChatTemplateContentFormatOption,
                                         ConversationMessage)
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (
    ChatCompletionAudio, ChatCompletionModality,
    ChatCompletionNamedToolChoiceParam, ChatCompletionRequest,
    ChatCompletionResponse, ChatCompletionResponseChoice, ChatMessage,
    ErrorResponse, FunctionCall, FunctionDefinition, PromptTokenUsageInfo,
    RequestResponseMetadata, ToolCall, UsageInfo)
from vllm.entrypoints.openai.serving_engine import (OpenAIServing,
                                                    clamp_prompt_logprobs)
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.entrypoints.openai.tool_parsers import ToolParser, ToolParserManager
from vllm.logger import init_logger
from vllm.model_executor.models.higgs_audio_tokenizer import (
    AudioTokenizer, revert_delay_pattern)
from vllm.outputs import RequestOutput
from vllm.reasoning import ReasoningParser, ReasoningParserManager
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils import random_uuid

logger = init_logger(__name__)


def convert_audio_to_base64(audio: np.ndarray,
                            sampling_rate: int,
                            target_format: str = "wav") -> str:
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, audio, sampling_rate, format=target_format)
    return base64.b64encode(audio_buffer.getvalue()).decode('utf-8')


class HiggsAudioServeEngine(OpenAIServing):

    def __init__(
        self,
        engine_client: EngineClient,
        model_config: ModelConfig,
        models: OpenAIServingModels,
        response_role: str,
        *,
        request_logger: Optional[RequestLogger],
        # chat_template: Optional[str],
        chat_template_content_format: ChatTemplateContentFormatOption,
        return_tokens_as_token_ids: bool = False,
        enable_reasoning: bool = False,
        reasoning_parser: Optional[str] = None,
        enable_auto_tools: bool = False,
        tool_parser: Optional[str] = None,
        enable_prompt_tokens_details: bool = False,
        audio_tokenizer: Optional[AudioTokenizer] = None,
    ):
        super().__init__(engine_client=engine_client,
                         model_config=model_config,
                         models=models,
                         request_logger=request_logger,
                         return_tokens_as_token_ids=return_tokens_as_token_ids)

        # Add a lock for generation
        self.request_logger = request_logger
        self.response_role = response_role
        # self.chat_template = chat_template
        self.chat_template_content_format: Final = chat_template_content_format
        self.enable_reasoning = enable_reasoning
        self.reasoning_parser = reasoning_parser

        # set up tool use
        self.enable_auto_tools: bool = enable_auto_tools
        if self.enable_auto_tools:
            logger.info(
                "\"auto\" tool choice has been enabled please note that while"
                " the parallel_tool_calls client option is preset for "
                "compatibility reasons, it will be ignored.")

        self.enable_reasoning: bool = enable_reasoning
        self.reasoning_parser: Optional[Callable[[AnyTokenizer],
                                                 ReasoningParser]] = None
        if self.enable_reasoning:
            try:
                self.reasoning_parser = (
                    ReasoningParserManager.get_reasoning_parser(
                        reasoning_parser))
            except Exception as e:
                raise TypeError("Error: --enable-reasoning requires "
                                f"reasoning_parser:'{reasoning_parser}' "
                                "which has not been registered") from e

        self.tool_parser: Optional[Callable[[AnyTokenizer], ToolParser]] = None
        if self.enable_auto_tools:
            try:
                if (tool_parser == "pythonic" and
                        model_config.model.startswith("meta-llama/Llama-3.2")):
                    logger.warning(
                        "Llama3.2 models may struggle to emit valid pythonic"
                        " tool calls")
                self.tool_parser = ToolParserManager.get_tool_parser(
                    tool_parser)
            except Exception as e:
                raise TypeError("Error: --enable-auto-tool-choice requires "
                                f"tool_parser:'{tool_parser}' which has not "
                                "been registered") from e

        self.enable_prompt_tokens_details = enable_prompt_tokens_details
        self.enable_prompt_tokens_details = enable_prompt_tokens_details
        self.default_sampling_params = (
            self.model_config.get_diff_sampling_param())
        if self.default_sampling_params:
            source = self.model_config.generation_config
            source = "model" if source == "auto" else source
            logger.info("Using default chat sampling params from %s: %s",
                        source, self.default_sampling_params)

        self.audio_tokenizer = audio_tokenizer
        self.audio_num_codebooks = self.audio_tokenizer.num_codebooks
        self.audio_codebook_size = self.audio_tokenizer.codebook_size
        self.audio_tokenizer_tps = self.audio_tokenizer.tps
        self.samples_per_token = int(self.audio_tokenizer.sampling_rate //
                                     self.audio_tokenizer_tps)
        self.hamming_window_len = \
            2 * self.audio_num_codebooks * self.samples_per_token

    # ruff: noqa: E501  # Disable specific lint rules
    def get_chat_template(
            self, modalities: Optional[list[ChatCompletionModality]]) -> str:
        if modalities is not None and "audio" in modalities:
            # fmt: off
            return (
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
                        "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n<|audio_out_bos|><|AUDIO_OUT|>' }}"
                    "{% endif %}"
                )
            # fmt: on

        # fmt: off
        return (
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
        # fmt: on

    async def create_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Optional[Request] = None,
    ) -> Union[AsyncGenerator[str, None], ChatCompletionResponse,
               ErrorResponse]:
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            logger.error("Error with model %s", error_check_ret)
            return error_check_ret

        if self.engine_client.errored:
            raise self.engine_client.dead_error

        try:
            (
                lora_request,
                prompt_adapter_request,
            ) = self._maybe_get_adapters(request)

            model_name = self._get_model_name(request.model, lora_request)

            tokenizer = await self.engine_client.get_tokenizer(lora_request)

            tool_parser = self.tool_parser

            if (request.tool_choice == "auto" and
                    not (self.enable_auto_tools and tool_parser is not None)):
                # for hf tokenizers, "auto" tools requires
                # --enable-auto-tool-choice and --tool-call-parser
                return self.create_error_response(
                    "\"auto\" tool choice requires "
                    "--enable-auto-tool-choice and --tool-call-parser to be set"
                )

            tool_dicts = None if request.tools is None else [
                tool.model_dump() for tool in request.tools
            ]

            chat_template = request.chat_template or \
                self.get_chat_template(request.modalities)
            (
                conversation,
                request_prompts,
                engine_prompts,
            ) = await self._preprocess_chat(
                request,
                tokenizer,
                request.messages,
                chat_template=chat_template,
                chat_template_content_format=self.chat_template_content_format,
                add_generation_prompt=request.add_generation_prompt,
                continue_final_message=request.continue_final_message,
                tool_dicts=tool_dicts,
                documents=request.documents,
                chat_template_kwargs=request.chat_template_kwargs,
                tool_parser=tool_parser,
                truncate_prompt_tokens=request.truncate_prompt_tokens,
                add_special_tokens=request.add_special_tokens,
            )
        except (ValueError, TypeError, RuntimeError,
                jinja2.TemplateError) as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))

        request_id = "chatcmpl-" \
                     f"{self._base_request_id(raw_request, request.request_id)}"
        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        # Schedule the request and get the result generator.
        generators: list[AsyncGenerator[RequestOutput, None]] = []
        try:
            for i, engine_prompt in enumerate(engine_prompts):
                sampling_params: Union[SamplingParams, BeamSearchParams]
                default_max_tokens = self.max_model_len - len(
                    engine_prompt["prompt_token_ids"])
                if request.use_beam_search:
                    sampling_params = request.to_beam_search_params(
                        default_max_tokens, self.default_sampling_params)
                else:
                    sampling_params = request.to_sampling_params(
                        default_max_tokens,
                        self.model_config.logits_processor_pattern,
                        self.default_sampling_params)

                self._log_inputs(request_id,
                                 request_prompts[i],
                                 params=sampling_params,
                                 lora_request=lora_request,
                                 prompt_adapter_request=prompt_adapter_request)

                trace_headers = (None if raw_request is None else await
                                 self._get_trace_headers(raw_request.headers))

                if isinstance(sampling_params, BeamSearchParams):
                    generator = self.engine_client.beam_search(
                        prompt=engine_prompt,
                        request_id=request_id,
                        params=sampling_params,
                    )
                else:
                    generator = self.engine_client.generate(
                        engine_prompt,
                        sampling_params,
                        request_id,
                        lora_request=lora_request,
                        trace_headers=trace_headers,
                        prompt_adapter_request=prompt_adapter_request,
                        priority=request.priority,
                    )

                generators.append(generator)
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        assert len(generators) == 1
        result_generator, = generators

        # Streaming response
        if request.stream:
            return self.chat_completion_stream_generator(
                request, result_generator, request_id, model_name,
                conversation, tokenizer, request_metadata)

        try:
            return await self.chat_completion_full_generator(
                request, result_generator, request_id, model_name,
                conversation, tokenizer, request_metadata)
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

    def get_chat_request_role(self, request: ChatCompletionRequest) -> str:
        if request.add_generation_prompt:
            return self.response_role
        return request.messages[-1]["role"]

    async def chat_completion_full_generator(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        model_name: str,
        conversation: list[ConversationMessage],
        tokenizer: AnyTokenizer,
        request_metadata: RequestResponseMetadata,
    ) -> Union[ErrorResponse, ChatCompletionResponse]:

        created_time = int(time.time())
        final_res: Optional[RequestOutput] = None

        try:
            async for res in result_generator:
                final_res = res
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        assert final_res is not None

        choices: list[ChatCompletionResponseChoice] = []

        role = self.get_chat_request_role(request)
        for output in final_res.outputs:
            token_ids = output.token_ids
            out_logprobs = output.logprobs

            if request.logprobs and request.top_logprobs is not None:
                assert out_logprobs is not None, "Did not output logprobs"
                logprobs = self._create_chat_logprobs(
                    token_ids=token_ids,
                    top_logprobs=out_logprobs,
                    num_output_top_logprobs=request.top_logprobs,
                    tokenizer=tokenizer,
                    return_as_token_id=request.return_tokens_as_token_ids,
                )
            else:
                logprobs = None

            should_stream_with_reasoning_parsing = (
                self._should_stream_with_reasoning_parsing(request))

            # In the OpenAI API the finish_reason is "tools_called"
            # if the tool choice is auto and the model produced a tool
            # call. The same is not true for named function calls
            auto_tools_called = False

            if should_stream_with_reasoning_parsing and \
                self.reasoning_parser is not None:
                try:
                    reasoning_parser = self.reasoning_parser(tokenizer)
                except RuntimeError as e:
                    logger.exception("Error in reasoning parser creation.")
                    return self.create_error_response(str(e))
                # If the reasoning parser is enabled,
                # tool calls are extracted exclusively from the content.
                reasoning_content, content = (
                    reasoning_parser.extract_reasoning_content(
                        output.text, request=request))
            else:
                reasoning_content = None
                content = output.text

            mm_token_ids = np.array(output.mm_token_ids, dtype=np.int64)

            # Post-process the audio tokens to audio waveform
            if mm_token_ids is not None:
                wv_numpy, sampling_rate = self.audio_tokenizer.decode(
                    vq_code=revert_delay_pattern(
                        mm_token_ids.transpose(1, 0).clip(
                            0, self.audio_codebook_size - 1)[:, 1:-1]))
                # audio_pcm16 = (wv_numpy * np.iinfo(np.int16).max).astype(np.int16)

                # Convert audio to base64
                response_audio_format = "wav" if request.audio is None \
                                        else request.audio.format
                audio_base64 = convert_audio_to_base64(wv_numpy, sampling_rate,
                                                       response_audio_format)
            else:
                audio_base64 = None

            # if auto tools are not enabled, and a named tool choice using
            #   outlines is not being used
            if (not self.enable_auto_tools or not self.tool_parser) and \
                (not isinstance(request.tool_choice,
                                ChatCompletionNamedToolChoiceParam
                                ) and request.tool_choice != "required"):
                message = ChatMessage(
                    role=role,
                    reasoning_content=reasoning_content,
                    content=content,
                    audio=ChatCompletionAudio(id=f"audio-{random_uuid()}",
                                              data=audio_base64,
                                              expires_at=0,
                                              transcript=""),
                )

            # if the request uses tools and specified a tool choice
            elif request.tool_choice and type(
                    request.tool_choice) is ChatCompletionNamedToolChoiceParam:

                tool_call_class = ToolCall
                message = ChatMessage(
                    role=role,
                    reasoning_content=reasoning_content,
                    content="",
                    tool_calls=[
                        tool_call_class(function=FunctionCall(
                            name=request.tool_choice.function.name,
                            arguments=content))
                    ])

            elif request.tool_choice and request.tool_choice == "required":
                tool_call_class = ToolCall

                # the fields of FunctionDefinition are a superset of the
                # tool call outputs and can be used for parsing
                tool_calls = TypeAdapter(
                    list[FunctionDefinition]).validate_json(output.text)
                message = ChatMessage(
                    role=role,
                    content="",
                    tool_calls=[
                        tool_call_class(function=FunctionCall(
                            name=tool_call.name,
                            arguments=json.dumps(tool_call.parameters)))
                        for tool_call in tool_calls
                    ])

            # if the request doesn't use tool choice
            # OR specifies to not use a tool
            elif not request.tool_choice or request.tool_choice == "none":

                message = ChatMessage(role=role,
                                      reasoning_content=reasoning_content,
                                      content=content,
                                      audio=ChatCompletionAudio(
                                          id=f"audio-{random_uuid()}",
                                          data=audio_base64,
                                          expires_at=0,
                                          transcript=""))

            # handle when there are tools and tool choice is auto
            elif request.tools and (
                    request.tool_choice == "auto"
                    or request.tool_choice is None) and self.enable_auto_tools \
                    and self.tool_parser:

                try:
                    tool_parser = self.tool_parser(tokenizer)
                except RuntimeError as e:
                    logger.exception("Error in tool parser creation.")
                    return self.create_error_response(str(e))

                tool_call_info = tool_parser.extract_tool_calls(
                    content if content is not None else "", request=request)
                # In the OpenAI API the finish_reason is "tools_called"
                # if the tool choice is auto and the model produced a tool
                # call. The same is not true for named function calls
                auto_tools_called = tool_call_info.tools_called
                if tool_call_info.tools_called:
                    message = ChatMessage(role=role,
                                          reasoning_content=reasoning_content,
                                          content=tool_call_info.content,
                                          tool_calls=tool_call_info.tool_calls)

                else:
                    # FOR NOW make it a chat message; we will have to detect
                    # the type to make it later.
                    message = ChatMessage(role=role,
                                          reasoning_content=reasoning_content,
                                          content=content)

            # undetermined case that is still important to handle
            else:
                logger.error(
                    "Error in chat_completion_full_generator - cannot determine"
                    " if tools should be extracted. Returning a standard chat "
                    "completion.")
                message = ChatMessage(role=role,
                                      reasoning_content=reasoning_content,
                                      content=content,
                                      audio=ChatCompletionAudio(
                                          id=f"audio-{random_uuid()}",
                                          data=audio_base64,
                                          expires_at=0,
                                          transcript=""))

            choice_data = ChatCompletionResponseChoice(
                index=output.index,
                message=message,
                logprobs=logprobs,
                finish_reason="tool_calls" if auto_tools_called else
                output.finish_reason if output.finish_reason else "stop",
                stop_reason=output.stop_reason)
            choices.append(choice_data)

        if request.echo:
            last_msg_content: Union[str, list[dict[str, str]]] = ""
            if conversation and "content" in conversation[-1] and conversation[
                    -1].get("role") == role:
                last_msg_content = conversation[-1]["content"] or ""
            if isinstance(last_msg_content, list):
                last_msg_content = "\n".join(msg['text']
                                             for msg in last_msg_content)

            for choice in choices:
                full_message = last_msg_content + (choice.message.content
                                                   or "")
                choice.message.content = full_message

        assert final_res.prompt_token_ids is not None
        num_prompt_tokens = len(final_res.prompt_token_ids)
        if final_res.encoder_prompt_token_ids is not None:
            num_prompt_tokens += len(final_res.encoder_prompt_token_ids)
        num_generated_tokens = sum(
            len(output.token_ids) for output in final_res.outputs)
        usage = UsageInfo(prompt_tokens=num_prompt_tokens,
                          completion_tokens=num_generated_tokens,
                          total_tokens=num_prompt_tokens +
                          num_generated_tokens)
        if self.enable_prompt_tokens_details and final_res.num_cached_tokens:
            usage.prompt_tokens_details = PromptTokenUsageInfo(
                cached_tokens=final_res.num_cached_tokens)

        request_metadata.final_usage_info = usage

        response = ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
            prompt_logprobs=clamp_prompt_logprobs(final_res.prompt_logprobs),
        )

        return response

    def _should_stream_with_reasoning_parsing(self,
                                              request: ChatCompletionRequest):
        """
            Utility function to check if streamed tokens should go through the
            reasoning parser that was configured.
    
            We only want to do this IF reasoning is enabled and a reasoning 
            parser is configured.
            """
        return self.enable_reasoning and self.reasoning_parser is not None
