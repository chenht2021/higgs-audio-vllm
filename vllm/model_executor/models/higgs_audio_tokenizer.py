# SPDX-License-Identifier: Apache-2.0
# ruff: noqa
"""Wrapper for audio tokenization."""
import os
import tempfile
import warnings
from enum import Enum
from functools import cache
from typing import Optional, Union

import boto3
import librosa
import numpy as np
import s3fs
import torch
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, AutoModel, AutoProcessor

MODEL_INFO = {
    "dac_16k": {
        "path": "descript/dac_16khz",
        "tps": 16000 / 320,
    },
    "dac_44k": {
        "path": "descript/dac_44khz",
        "tps": 44100 / 512,
    },
    "mimi": {
        "path": "kyutai/mimi",
        "tps": 12.5,
    },
    # For xcodec, we save the config file to "base_folder/config.yaml" and the model file to "base_folder/model.pth"
    "xcodec_tps50": {
        "path": "s3://data/audio_tokenizer/xcodec_tps50/original",
        "tps": 50,
        "sampling_rate": 16000,
        "num_codebooks": 8,
        "codebook_size": 1024,
    },
    "xcodec_tps25_0215": {
        "path": "s3://data/audio_tokenizer/xcodec_tps25/0215_exp1_step630k",
        "tps": 25,
        "sampling_rate": 16000,
        "num_codebooks": 8,
        "codebook_size": 1024,
    },

    # For xcodec2, we save the pt file directly
    "xcodec2_tps50": {
        "path":
        "s3://data/audio_tokenizer/xcodec2_tps50/original/epoch=4-step=1400000_weights.pt",
        "tps": 50,
        "sampling_rate": 16000,
        "num_codebooks": 1,
        "codebook_size": 65536,  # 4**8
    },
    "xcodec2_tps50_0223": {
        "path":
        "s3://data/audio_tokenizer/xcodec2_tps50/0223/epoch=1-step=240000_weights.pt",
        "tps": 50,
        "sampling_rate": 16000,
        "num_codebooks": 1,
        "codebook_size": 65536,  # 4**8
    },
    "cosyvoice2": {
        "path": "s3://data/audio_tokenizer/cosyvoice2",
        "tps": 25,
        "sampling_rate":
        16000,  # CosyVoice2's input uses 16000 sampling rate but the flow + hifigan model uses 24000
        "num_codebooks": 1,
        "codebook_size": 6561,
    }
}


def connect_to_s3(s3_key=None, s3_secret=None, profile_name: str = "default"):
    """Connect to S3."""
    endpoint_url = "http://s3.canada.boson.ai" if profile_name == "ceph" else "http://totoro.canada.boson.ai"

    if s3_key is None or s3_secret is None:
        session = boto3.Session(profile_name=profile_name)
        s3_key = session.get_credentials().access_key
        s3_secret = session.get_credentials().secret_key

    s3 = s3fs.S3FileSystem(endpoint_url=endpoint_url,
                           key=s3_key,
                           secret=s3_secret)
    return s3


def revert_delay_pattern(data):
    """Convert samples encoded with delay pattern back to the original form.

    Args:
        data (:obj:`torch.Tensor`):
            The data with delay pattern applied. It will have shape (num_codebooks, seq_len + num_codebooks - 1).

    Returns:
        ret (:obj:`torch.Tensor`):
            Recovered data with delay pattern removed. It will have shape (num_codebooks, seq_len).
    """
    assert len(data.shape) == 2
    out_l = []
    num_codebooks = data.shape[0]
    for i in range(num_codebooks):
        out_l.append(data[i:(i + 1),
                          i:(data.shape[1] - num_codebooks + 1 + i)])
    return np.concatenate(out_l, axis=0)


class AudioTokenizerType(Enum):
    """Enum for audio tokenizers."""
    DAC = "dac"
    MIMI = "mimi"
    XCODEC = "xcodec"
    XCODEC2 = "xcodec2"
    COSYVOICE2 = "cosyvoice2"


# Brought from https://github.com/openai/whisper/blob/517a43ecd132a2089d85f4ebc044728a71d49f6e/whisper/audio.py#L91-L92
@cache
def mel_filters(device, n_mels: int) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
            mel_128=librosa.filters.mel(sr=16000, n_fft=400, n_mels=128),
        )
    """
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"

    filters_path = os.path.join(os.path.dirname(__file__), "assets",
                                "mel_filters.npz")
    with np.load(filters_path, allow_pickle=False) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


# Brought from https://github.com/openai/whisper/blob/517a43ecd132a2089d85f4ebc044728a71d49f6e/whisper/audio.py#L110
# We added N_FFT, mel_filters
def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int = 80,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
    n_fft: int = 400,
    hop_length: int = 160,
):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 and 128 are supported

    padding: int
        Number of zero samples to pad to the right

    device: Optional[Union[str, torch.device]]
        If given, the audio tensor is moved to this device before STFT

    Returns
    -------
    torch.Tensor, shape = (n_mels, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio, _ = librosa.load(audio, mono=True, sr=16000)
        audio = torch.from_numpy(audio)

    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(n_fft).to(audio.device)
    stft = torch.stft(audio,
                      n_fft,
                      hop_length,
                      window=window,
                      return_complex=True)
    magnitudes = stft[..., :-1].abs()**2

    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


class AudioTokenizer:
    """Common interface for audio tokenizers. We support the following tokenizers
 
    - dac
    - mimi
    - xcodec
    - xcodec2
    - cosyvoice2

    """

    def __init__(self,
                 model,
                 device="cuda:0",
                 compile=False,
                 downloaded_model_path=None):
        self._model = model
        self._device = device

        model_path = downloaded_model_path or MODEL_INFO[model]["path"]
        self._tps = MODEL_INFO[model]["tps"]
        if model == "mimi" or model.startswith("dac"):
            self.audio_tokenizer_model = AutoModel.from_pretrained(
                model_path, trust_remote_code=True)
            self.audio_tokenizer_feature_extractor = AutoFeatureExtractor.from_pretrained(
                model_path)

            if model.startswith("dac"):
                self.model_type = AudioTokenizerType.DAC
                self._num_codebooks = self.audio_tokenizer_model.config.n_codebooks
            else:
                self.model_type = AudioTokenizerType.MIMI
                self._num_codebooks = self.audio_tokenizer_model.config.num_quantizers

            self._codebook_size = self.audio_tokenizer_model.config.codebook_size
            self._sampling_rate = AutoProcessor.from_pretrained(
                model_path).sampling_rate

        elif model.startswith("xcodec_"):
            from xcodec.xcodec_wrapper import (XCodecFeatureExtractor,
                                               load_codec_model)
            self.model_type = AudioTokenizerType.XCODEC

            if model_path.startswith("s3://"):
                s3 = connect_to_s3()
                with tempfile.TemporaryDirectory() as tmp_dir:
                    s3.download(f"{model_path}/config.yaml",
                                os.path.join(tmp_dir, "config.yaml"))
                    s3.download(f"{model_path}/model.pth",
                                os.path.join(tmp_dir, "model.pth"))

                    self.audio_tokenizer_model = load_codec_model(
                        os.path.join(tmp_dir, "config.yaml"),
                        os.path.join(tmp_dir, "model.pth"))
            else:
                self.audio_tokenizer_model = load_codec_model(
                    os.path.join(model_path, "config.yaml"),
                    os.path.join(model_path, "model.pth"))
            self._sampling_rate = MODEL_INFO[model]["sampling_rate"]
            self.audio_tokenizer_feature_extractor = XCodecFeatureExtractor(
                sampling_rate=self._sampling_rate)
            self._num_codebooks = MODEL_INFO[model]["num_codebooks"]
            self._codebook_size = MODEL_INFO[model]["codebook_size"]
        elif model.startswith("xcodec2_"):
            from xcodec2_inference.configuration_bigcodec import BigCodecConfig
            from xcodec2_inference.modeling_xcodec2 import XCodec2Model

            config = BigCodecConfig()
            self.audio_tokenizer_model = XCodec2Model(config)

            if model_path.startswith("s3://"):
                s3 = connect_to_s3()
                with tempfile.TemporaryDirectory() as tmp_dir:
                    s3.download(model_path,
                                os.path.join(tmp_dir, "weights.pt"))
                    state_dict = torch.load(os.path.join(
                        tmp_dir, "weights.pt"),
                                            map_location="cpu")

            else:
                state_dict = torch.load(os.path.join(model_path, "weights.pt"),
                                        map_location="cpu")

            self.audio_tokenizer_model.load_state_dict(state_dict)
            self.audio_tokenizer_feature_extractor = None
            self.model_type = AudioTokenizerType.XCODEC2
            self._sampling_rate = MODEL_INFO[model]["sampling_rate"]
            self._num_codebooks = MODEL_INFO[model]["num_codebooks"]
            self._codebook_size = MODEL_INFO[model]["codebook_size"]
        elif model == "cosyvoice2":
            import onnxruntime
            from cosyvoice.tokenizer.tokenizer import QwenTokenizer
            from hyperpyyaml import load_hyperpyyaml

            if not model_path.startswith("s3://"):
                raise ValueError(
                    f"Cosoyvoice model path {model_path} must start with s3://"
                )

            self.model_type = AudioTokenizerType.COSYVOICE2
            self._sampling_rate = MODEL_INFO[model]["sampling_rate"]
            s3 = connect_to_s3()

            with tempfile.TemporaryDirectory() as tmp_dir:
                os.makedirs(os.path.join(tmp_dir, "tokenizer"), exist_ok=True)
                s3.download(
                    os.path.join(model_path, "speech_tokenizer_v2.onnx"),
                    os.path.join(tmp_dir, "speech_tokenizer_v2.onnx"))
                for file in [
                        "speech_tokenizer_v2.onnx", "cosyvoice_rev.yaml",
                        "flow.pt", "hift.pt", "campplus.onnx",
                        "tokenizer/tokenizer_config.json",
                        "tokenizer/vocab.json", "tokenizer/merges.txt"
                ]:
                    s3.download(os.path.join(model_path, file),
                                os.path.join(tmp_dir, file))
                with open(os.path.join(tmp_dir, "cosyvoice_rev.yaml")) as f:
                    self._config = load_hyperpyyaml(f)
                self._text_tokenizer = QwenTokenizer(
                    os.path.join(tmp_dir, "tokenizer"))

                self._feat_extractor = self._config["feat_extractor"]
                self._flow = self._config["flow"]
                self._hift = self._config["hift"]
                self._flow.load_state_dict(torch.load(
                    os.path.join(tmp_dir, "flow.pt"),
                    map_location=self._device),
                                           strict=True)
                self._flow.to(self._device).eval()
                self._flow.fp16 = False
                self._flow.encoder.static_chunk_size = 2 * self._flow.input_frame_rate
                self._flow.decoder.estimator.static_chunk_size = 2 * self._flow.input_frame_rate * self._flow.token_mel_ratio
                # in case hift_model is a hifigan model
                hift_state_dict = {
                    k.replace('generator.', ''): v
                    for k, v in torch.load(os.path.join(tmp_dir, "hift.pt"),
                                           map_location=self._device).items()
                }
                self._hift.load_state_dict(hift_state_dict, strict=True)
                self._hift.to(self._device).eval()

                option = onnxruntime.SessionOptions()
                option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
                option.intra_op_num_threads = 1
                speech_tokenizer_model = os.path.join(
                    tmp_dir, "speech_tokenizer_v2.onnx")

                self._campplus_session = onnxruntime.InferenceSession(
                    os.path.join(tmp_dir, "campplus.onnx"),
                    sess_options=option,
                    providers=["CPUExecutionProvider"])
                self.audio_tokenizer_model = onnxruntime.InferenceSession(
                    speech_tokenizer_model,
                    sess_options=option,
                    providers=[
                        "CUDAExecutionProvider" if torch.cuda.is_available()
                        and "cuda" in str(device) else "CPUExecutionProvider"
                    ])
                self.audio_tokenizer_feature_extractor = None

        else:
            raise ValueError(f"Unsupported audio tokenizer {model}")

        if self.model_type != AudioTokenizerType.COSYVOICE2:
            self.audio_tokenizer_model.eval()
            self.audio_tokenizer_model.to(device)
            if compile:
                self.audio_tokenizer_model = torch.compile(
                    self.audio_tokenizer_model)

    def _extract_cosyvoice2_spk_embedding(self, speech):
        """Extract the speaker embedding from the speech.
        
        Function is based on https://github.com/FunAudioLLM/CosyVoice/blob/fd45708e4beb6ae40d1344452d7010cc338b0768/cosyvoice/cli/frontend.py#L104

        """
        import torchaudio.compliance.kaldi as kaldi

        feat = kaldi.fbank(speech,
                           num_mel_bins=80,
                           dither=0,
                           sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = self._campplus_session.run(
            None, {
                self._campplus_session.get_inputs()[0].name:
                feat.unsqueeze(dim=0).cpu().numpy()
            })[0].flatten().tolist()
        embedding = torch.tensor([embedding]).to(self._device)
        return embedding

    @property
    def tps(self):
        return self._tps

    @property
    def sampling_rate(self):
        return self._sampling_rate

    @property
    def num_codebooks(self):
        return self._num_codebooks

    @property
    def codebook_size(self):
        return self._codebook_size

    @property
    def tps(self):
        return self._tps

    def encode(self,
               audio_path_or_wv,
               sr=None,
               loudness_normalize=False,
               loudness_threshold=-23.0):
        if isinstance(audio_path_or_wv, str):
            wv, sr = librosa.load(audio_path_or_wv, mono=True, sr=None)
        else:
            wv = audio_path_or_wv
            assert sr is not None
        if loudness_normalize:
            import pyloudnorm as pyln
            meter = pyln.Meter(sr)
            l = meter.integrated_loudness(wv)
            wv = pyln.normalize.loudness(wv, l, loudness_threshold)
        if sr != self.sampling_rate:
            wv = librosa.resample(wv, orig_sr=sr, target_sr=self.sampling_rate)
        if self.audio_tokenizer_feature_extractor is not None:
            inputs = self.audio_tokenizer_feature_extractor(
                raw_audio=wv,
                sampling_rate=self.audio_tokenizer_feature_extractor.
                sampling_rate,
                return_tensors="pt")
            input_values = inputs["input_values"].to(self._device)
        else:
            input_values = torch.from_numpy(wv).float().unsqueeze(0)
        with torch.no_grad():
            if self.model_type in [
                    AudioTokenizerType.DAC, AudioTokenizerType.MIMI,
                    AudioTokenizerType.XCODEC
            ]:
                encoder_outputs = self.audio_tokenizer_model.encode(
                    input_values)
                vq_code = encoder_outputs.audio_codes[0]
            elif self.model_type == AudioTokenizerType.COSYVOICE2:
                feat = log_mel_spectrogram(input_values, n_mels=128)
                speech_token = self.audio_tokenizer_model.run(
                    None, {
                        self.audio_tokenizer_model.get_inputs()[0].name:
                        feat.detach().cpu().numpy(),
                        self.audio_tokenizer_model.get_inputs()[1].name:
                        np.array([feat.shape[2]], dtype=np.int32)
                    })[0].flatten().tolist()
                vq_code = torch.tensor([speech_token], dtype=torch.int32)
            elif self.model_type == AudioTokenizerType.XCODEC2:
                vq_code = self.audio_tokenizer_model.encode_code(
                    input_waveform=input_values)[0]
            else:
                raise NotImplementedError(
                    f"Audio tokenizer {self.model_type} not implemented")
        return vq_code

    def decode(self,
               vq_code,
               prompt_audio=None,
               prompt_audio_sr=None,
               return_cuda_tensor=False):
        """Decode the audio codes to waveform.
        
        Parameters:
        -----------
        vq_code: torch.Tensor
            The audio codes to decode. Shape (num_codebooks, total_length)
        prompt_audio
            This flag is only used in CosyVoice2! It contains the prompt audio corresponding to the text.
        prompt_audio_sr
            This flag is only used in CosyVoice2! It contains the sample rate of the prompt audio.

        Returns:
        --------
        decoded_wv: np.ndarray
            The decoded waveform. Shape (#time,)
        sampling_rate: int
            The sampling rate of the decoded waveform.
        """
        with torch.no_grad():
            if isinstance(vq_code, torch.Tensor):
                vq_code = vq_code.to(self._device)
            else:
                vq_code = torch.from_numpy(vq_code).to(self._device)
            if self.model_type == AudioTokenizerType.DAC:
                decoded_wv = self.audio_tokenizer_model.decode(
                    audio_codes=vq_code.unsqueeze(0)).audio_values[0]
            elif self.model_type == AudioTokenizerType.MIMI:
                decoded_wv = self.audio_tokenizer_model.decode(
                    audio_codes=vq_code.unsqueeze(0)).audio_values[0, 0]
            elif self.model_type == AudioTokenizerType.XCODEC:
                decoded_wv = self.audio_tokenizer_model.decode(
                    codes=vq_code.unsqueeze(0))[0, 0]
            elif self.model_type == AudioTokenizerType.XCODEC2:
                decoded_wv = self.audio_tokenizer_model.decode_code(
                    vq_code.unsqueeze(0))[0, 0]
            elif self.model_type == AudioTokenizerType.COSYVOICE2:
                if prompt_audio is None:
                    warnings.warn(
                        "Prompt text is not provided. We will use an empty prompt. The generated audio may lose the speaker information."
                    )
                    flow_embedding = torch.zeros(
                        (1, 192), dtype=torch.float32).to(self._device)
                    prompt_token = torch.zeros(
                        (1, 0), dtype=torch.int32).to(self._device)
                    prompt_feat = torch.zeros(
                        (1, 0, 80), dtype=torch.float32).to(self._device)
                else:
                    prompt_token = self.encode(prompt_audio,
                                               sr=prompt_audio_sr).to(
                                                   self._device)
                    if isinstance(prompt_audio, str):
                        wv_16k, _ = librosa.load(prompt_audio,
                                                 mono=True,
                                                 sr=16000)
                    else:
                        if prompt_audio_sr != 16000:
                            wv_16k = librosa.resample(prompt_audio,
                                                      orig_sr=prompt_audio_sr,
                                                      target_sr=24000)
                        else:
                            wv_16k = prompt_audio
                    wv_24k = librosa.resample(wv_16k,
                                              orig_sr=16000,
                                              target_sr=24000)
                    wv_24k = torch.from_numpy(wv_24k).float().unsqueeze(0)
                    wv_16k = torch.from_numpy(wv_16k).float().unsqueeze(0)

                    flow_embedding = self._extract_cosyvoice2_spk_embedding(
                        wv_16k)
                    prompt_feat = self._feat_extractor(wv_24k).squeeze(
                        dim=0).transpose(0, 1).to(self._device)
                    prompt_feat = prompt_feat.unsqueeze(dim=0)
                with torch.no_grad():
                    tts_mel, _ = self._flow.inference(
                        token=vq_code.to(self._device),
                        token_len=torch.tensor([vq_code.shape[1]],
                                               dtype=torch.int32).to(
                                                   self._device),
                        prompt_token=prompt_token,
                        prompt_token_len=torch.tensor([prompt_token.shape[1]],
                                                      dtype=torch.int32).to(
                                                          self._device),
                        prompt_feat=prompt_feat.to(self._device),
                        prompt_feat_len=torch.tensor([prompt_feat.shape[1]],
                                                     dtype=torch.int32).to(
                                                         self._device),
                        embedding=flow_embedding.to(self._device),
                        finalize=True)
                    decoded_wv, _ = self._hift.inference(speech_feat=tts_mel)
                    decoded_wv = decoded_wv[0]
            else:
                raise NotImplementedError(
                    f"Audio tokenizer {self.model_type} not implemented")

            if not return_cuda_tensor:
                if decoded_wv.device != "cpu":
                    decoded_wv = decoded_wv.cpu()
                decoded_wv = decoded_wv.numpy()

            if self.model_type == AudioTokenizerType.COSYVOICE2:
                sampling_rate = 24000
            else:
                sampling_rate = self.sampling_rate
            return decoded_wv, sampling_rate
