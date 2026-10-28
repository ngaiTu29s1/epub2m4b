"""RVC v2 (Retrieval-based Voice Conversion) inference, Python 3.12 compatible.

The upstream `rvc-python` package pins ``fairseq==0.12.2`` and ``numpy<1.26``,
both of which fail to build on Python 3.12 (``distutils`` removal). This module
re-implements the *inference* half of RVC v2 in pure PyTorch / transformers so
it runs on the repo's existing ``.venv`` with no extra native toolchain.

Pipeline (input -> output)::

    Edge-TTS 24 kHz wav
      -> resample to 16 kHz, run ContentVec (HuBERT 768-dim)      -> features
      -> extract F0 with RMVPE, map to the model's pitch bins      -> f0
      -> optional FAISS retrieval: blend features toward the target index
      -> RVC v2 generator (SynthesizerTrnMs768NSFsid) + NSF-HiFiGAN
      -> resample back to 24 kHz, write mono PCM_16 wav

The heavy pieces (torch, transformers, faiss, the checkpoints) are imported
lazily so merely importing this module stays cheap and side-effect free.
"""
from __future__ import annotations

import logging
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger('epub2m4b.rvc')

SAMPLE_RATE = 24000
RVC_SAMPLE_RATE = 40000
RVC_VERSION = 'v2'
HUBERT_FEATURE_DIM = 768

# RVC's internal signal geometry (see rvc-python modules/vc/pipeline.py).
RVC_WINDOW = 160          # samples per F0/hop step at 16 kHz => 100 steps/sec
RVC_FEATURE_SR = 16000    # ContentVec / F0 working sample rate
RVC_F0_MIN = 50.0
RVC_F0_MAX = 1100.0

PRETRAINED_DIR = Path('models/rvc_pretrained')
RVC_GENERATOR_FILE = 'pretrained_v2/f0G40k.pth'
CONTENTVEC_REPO = 'lengyue233/content-vec-best'
HUBERT_BASE_REPO = 'lj1995/VoiceConversionWebUI'
HUBERT_BASE_FILE = 'hubert_base.pt'
RMVPE_FILE = 'rmvpe.pt'

DEFAULT_MODEL_DIR_NAME = 'models/rvc_vui_ve'

# RVC v2 40k generator topology, read off pretrained_v2/f0G40k.pth.
_INTER_CHANNELS = 192
_SPEAKER_EMBED_DIM = 256
_HIDDEN_CHANNELS = 192
_GIN_CHANNELS = 256
_FILTER_CHANNELS = 768
_N_HEADS = 2
_WINDOW_SIZE = 10
_UPSAMPLE_INITIAL = 512
_UPSAMPLE_RATES = [10, 10, 2, 2]
_UPSAMPLE_KERNELS = [16, 16, 4, 4]
_RESBLOCK_KERNELS = [3, 7, 11]
_RESBLOCK_DILATIONS = [
    [1, 3, 5],
    [1, 3, 5],
    [1, 3, 5],
]
_N_FFT = 2048
_HOP_LENGTH = 480
_WIN_LENGTH = 2048
_NUM_MELS = 80
_N_SPEAKERS = 109


def expected_artifacts(model_dir: Path, voice_name: str) -> dict[str, Path]:
    """Conventional on-disk paths for a trained RVC voice."""
    model_dir = Path(model_dir)
    return {
        'model': model_dir / f'{voice_name}.pth',
        'index': model_dir / f'{voice_name}.index',
        'dir': model_dir,
    }


def build_feature_index(converter, wav_dir, out_index):
    """Extract ContentVec features from every wav in `wav_dir` into a FAISS index.

    Returns the number of stored feature vectors. The index powers RVC's
    retrieval step, which nudges the ContentVec features toward the target
    speaker's timbre (``index_rate``) without retraining the generator.
    """
    import faiss
    import numpy as np
    import soundfile as sf

    wav_dir = Path(wav_dir)
    out_index = Path(out_index)
    out_index.parent.mkdir(parents=True, exist_ok=True)

    chunks = []
    wavs = sorted([p for p in wav_dir.glob('*.wav')
                   if not p.name.startswith('.')])
    for wav in wavs:
        audio, sr = sf.read(str(wav), dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        feats = converter._extract_features(audio, sr)
        if hasattr(feats, 'detach'):
            feats = feats.detach().cpu()
        feats = np.asarray(feats, dtype=np.float32)
        # (1, dim, frames) -> (frames, dim)
        arr = feats[0].T if feats.ndim == 3 else feats
        if arr.shape[0] == 0:
            continue
        chunks.append(arr)

    if not chunks:
        raise ValueError(f'Không tìm thấy wav nào trong {wav_dir}')

    all_feats = np.concatenate(chunks, axis=0).astype(np.float32)
    dim = all_feats.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(all_feats)
    faiss.write_index(index, str(out_index))
    logger.info(f'🔎 [RVC] Đã tạo FAISS index: {out_index} '
                f'({index.ntotal} vectors từ {len(wavs)} wav)')
    return int(index.ntotal)


# --------------------------------------------------------------------------- #
# Model architecture (RVC v2, SynthesizerTrnMs768NSFsid)
# --------------------------------------------------------------------------- #
def _build_models_classes():  # pragma: no cover - exercised via integration
    """Return the synthesizer class, defined lazily so import stays cheap."""
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.nn.utils import remove_weight_norm, weight_norm
    import numpy as np
    import math

    def convert_pad_shape(pad_shape):
        layer = pad_shape[::-1]
        return [item for sublist in layer for item in sublist]

    class LayerNorm(nn.Module):
        def __init__(self, channels, eps=1e-5):
            super().__init__()
            self.channels = channels
            self.eps = eps
            self.gamma = nn.Parameter(torch.ones(channels))
            self.beta = nn.Parameter(torch.zeros(channels))

        def forward(self, x):
            x = x.transpose(1, -1)
            x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
            return x.transpose(1, -1)

    class ConvReluNorm(nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels,
                     kernel_size, n_layers, p_dropout):
            super().__init__()
            self.in_channels = in_channels
            self.hidden_channels = hidden_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.n_layers = n_layers
            self.p_dropout = p_dropout
            convs = []
            for i in range(n_layers):
                i_in = in_channels if i == 0 else hidden_channels
                i_out = out_channels if i == n_layers - 1 else hidden_channels
                convs.append(weight_norm(nn.Conv1d(i_in, i_out, kernel_size,
                                                   padding=(kernel_size - 1) // 2)))
            self.convs = nn.ModuleList(convs)
            self.norms = nn.ModuleList(
                [LayerNorm(hidden_channels) for _ in range(n_layers - 1)]
            )
            self.relu = nn.ReLU()
            self.drop = nn.Dropout(p_dropout)

        def forward(self, x, x_mask):
            x_res = x
            x = self.relu(self.convs[0](x * x_mask))
            for i in range(self.n_layers - 1):
                x = self.drop(self.relu(self.norms[i](x)))
                x = self.convs[i + 1](x * x_mask)
            return (x + x_res) * x_mask

    class DDSConv(nn.Module):
        def __init__(self, channels, kernel_size, n_layers, p_dropout=0.0):
            super().__init__()
            self.channels = channels
            self.kernel_size = kernel_size
            self.n_layers = n_layers
            self.p_dropout = p_dropout
            self.drop = nn.Dropout(p_dropout)
            self.convs_sep = nn.ModuleList()
            self.convs_1x1 = nn.ModuleList()
            self.norms_1 = nn.ModuleList()
            self.norms_2 = nn.ModuleList()
            for i in range(n_layers):
                dilation = kernel_size ** i
                padding = (kernel_size * dilation - dilation) // 2
                self.convs_sep.append(
                    nn.Conv1d(channels, channels, kernel_size,
                              groups=channels, dilation=dilation, padding=padding)
                )
                self.convs_1x1.append(nn.Conv1d(channels, channels, 1))
                self.norms_1.append(LayerNorm(channels))
                self.norms_2.append(LayerNorm(channels))

        def forward(self, x, x_mask, g=None):
            if g is not None:
                x = x + g
            for i in range(self.n_layers):
                y = self.convs_sep[i](x * x_mask)
                y = self.norms_1[i](y)
                y = F.gelu(y)
                y = self.convs_1x1[i](y)
                y = self.norms_2[i](y)
                y = F.gelu(y)
                y = self.drop(y)
                x = x + y
            return x * x_mask

    class WN(nn.Module):
        def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers,
                     gin_channels):
            super().__init__()
            self.hidden_channels = hidden_channels
            self.kernel_size = kernel_size
            self.dilation_rate = dilation_rate
            self.n_layers = n_layers
            self.gin_channels = gin_channels
            self.in_layers = nn.ModuleList()
            self.res_skip_layers = nn.ModuleList()
            self.cond_layer = None
            if gin_channels != 0:
                self.cond_layer = weight_norm(
                    nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
                )
            for i in range(n_layers):
                dilation = dilation_rate ** i
                padding = int((kernel_size * dilation - dilation) / 2)
                self.in_layers.append(
                    weight_norm(nn.Conv1d(hidden_channels, 2 * hidden_channels,
                                          kernel_size, dilation=dilation, padding=padding))
                )
                res_skip_channels = (hidden_channels if i == n_layers - 1
                                     else 2 * hidden_channels)
                self.res_skip_layers.append(
                    weight_norm(nn.Conv1d(hidden_channels, res_skip_channels, 1))
                )

        def forward(self, x, x_mask, g=None):
            output = torch.zeros_like(x)
            if g is not None:
                g = self.cond_layer(g)
            for i in range(self.n_layers):
                x_in = self.in_layers[i](x)
                if g is not None:
                    cond_offset = i * 2 * self.hidden_channels
                    g_l = g[:, cond_offset:cond_offset + 2 * self.hidden_channels, :]
                    x_in = x_in + g_l
                acts = x_in
                acts1, acts2 = acts.chunk(2, 1)
                acts = torch.tanh(acts1) * torch.sigmoid(acts2)
                res_skip_acts = self.res_skip_layers[i](acts)
                if i < self.n_layers - 1:
                    res_acts = res_skip_acts[:, :self.hidden_channels, :]
                    x = (x + res_acts) * x_mask
                    output = output + res_skip_acts[:, self.hidden_channels:, :]
                else:
                    output = output + res_skip_acts
            return output * x_mask

    class MultiHeadAttention(nn.Module):
        def __init__(self, channels, out_channels, n_heads, window_size=None,
                     heads_share=True, block_length=None, proximal_bias=False,
                     proximal_init=False):
            super().__init__()
            self.channels = channels
            self.out_channels = out_channels
            self.n_heads = n_heads
            self.k_channels = channels // n_heads
            self.conv_q = nn.Conv1d(channels, channels, 1)
            self.conv_k = nn.Conv1d(channels, channels, 1)
            self.conv_v = nn.Conv1d(channels, channels, 1)
            self.conv_o = nn.Conv1d(channels, out_channels, 1)
            self.window_size = window_size
            # RVC's relative-position embeddings: one per head, windowed.
            n_heads_rel = 1 if heads_share else n_heads
            self.emb_rel_k = nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels)
                * (self.k_channels ** -0.5)
            )
            self.emb_rel_v = nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels)
                * (self.k_channels ** -0.5)
            )

        def _get_relative_embeddings(self, relative_embeddings, length):
            pad_length = max(length - (self.window_size + 1), 0)
            slice_start_position = max((self.window_size + 1) - length, 0)
            slice_end_position = slice_start_position + 2 * length - 1
            if pad_length > 0:
                padded_relative_embeddings = F.pad(
                    relative_embeddings,
                    convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]),
                )
            else:
                padded_relative_embeddings = relative_embeddings
            used_relative_embeddings = padded_relative_embeddings[
                :, slice_start_position:slice_end_position
            ]
            return used_relative_embeddings

        def _relative_position_to_absolute(self, x):
            batch_size, n_heads, length, _ = x.size()
            x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]]))
            x_flat = x.view([batch_size, n_heads, length * 2 * length])
            x_flat = F.pad(x_flat, convert_pad_shape([[0, 0], [0, 0], [0, length - 1]]))
            x_final = x_flat.view([batch_size, n_heads, length + 1, 2 * length - 1])[
                :, :, :length, length - 1:
            ]
            return x_final

        def _absolute_position_to_relative(self, x):
            batch_size, n_heads, length, _ = x.size()
            x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length - 1]]))
            x_flat = x.reshape([batch_size, n_heads, length ** 2 + length * (length - 1)])
            x_flat = F.pad(x_flat, convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
            x_final = x_flat.reshape(
                [batch_size, n_heads, length, 2 * length]
            )[:, :, :, 1:]
            return x_final

        def _matmul_with_relative_values(self, x, y):
            return torch.matmul(x, y.unsqueeze(0))

        def _matmul_with_relative_keys(self, x, y):
            return torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))

        def forward(self, x, x_mask, attn_mask=None):
            B, channels, T = x.size()
            q = self.conv_q(x) * x_mask
            k = self.conv_k(x) * x_mask
            v = self.conv_v(x) * x_mask

            query = q.view(B, self.n_heads, self.k_channels, T).transpose(2, 3)
            key = k.view(B, self.n_heads, self.k_channels, T).transpose(2, 3)
            value = v.view(B, self.n_heads, self.k_channels, T).transpose(2, 3)

            scores = torch.matmul(query / (self.k_channels ** 0.5),
                                  key.transpose(-2, -1))

            rel_k = self._get_relative_embeddings(self.emb_rel_k, T)
            rel_logits = self._matmul_with_relative_keys(
                query / (self.k_channels ** 0.5), rel_k)
            scores_local = self._relative_position_to_absolute(rel_logits)
            scores = scores + scores_local

            if attn_mask is not None:
                scores = scores.masked_fill(attn_mask == 0, -1e4)

            p_attn = F.softmax(scores, dim=-1)
            output = torch.matmul(p_attn, value)

            rel_v = self._get_relative_embeddings(self.emb_rel_v, T)
            rel_weights = self._absolute_position_to_relative(p_attn)
            rel_output = self._matmul_with_relative_values(rel_weights, rel_v)
            output = output + rel_output

            output = output.transpose(2, 3).contiguous().view(B, channels, T)
            output = self.conv_o(output)
            return output

    class FFN(nn.Module):
        def __init__(self, in_channels, out_channels, filter_channels,
                     kernel_size, p_dropout=0.0):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.filter_channels = filter_channels
            self.kernel_size = kernel_size
            self.p_dropout = p_dropout
            self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size,
                                    padding=kernel_size // 2)
            self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size,
                                    padding=kernel_size // 2)
            self.drop = nn.Dropout(p_dropout)

        def forward(self, x, x_mask):
            x = self.conv_1(x * x_mask)
            x = torch.relu(x)
            x = self.drop(x)
            x = self.conv_2(x * x_mask)
            return x * x_mask

    class Encoder(nn.Module):
        def __init__(self, hidden_channels, filter_channels, n_heads, n_layers,
                     kernel_size, p_dropout, window_size=10):
            super().__init__()
            self.hidden_channels = hidden_channels
            self.n_layers = n_layers
            self.attn_layers = nn.ModuleList()
            self.norm_layers_1 = nn.ModuleList()
            self.ffn_layers = nn.ModuleList()
            self.norm_layers_2 = nn.ModuleList()
            for i in range(n_layers):
                self.attn_layers.append(
                    MultiHeadAttention(hidden_channels, hidden_channels, n_heads,
                                       window_size=window_size)
                )
                self.norm_layers_1.append(LayerNorm(hidden_channels))
                self.ffn_layers.append(
                    FFN(hidden_channels, hidden_channels, filter_channels,
                        kernel_size, p_dropout)
                )
                self.norm_layers_2.append(LayerNorm(hidden_channels))

        def forward(self, x, x_mask):
            attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
            x = x * x_mask
            for i in range(self.n_layers):
                y = self.attn_layers[i](x, x_mask, attn_mask)
                x = self.norm_layers_1[i](x + y)
                y = self.ffn_layers[i](x, x_mask)
                x = self.norm_layers_2[i](x + y)
            return x * x_mask

    class TextEncoder(nn.Module):
        def __init__(self, out_channels, hidden_channels, filter_channels,
                     n_heads, n_layers, kernel_size, p_dropout,
                     gin_channels, spk_embed_dim):
            super().__init__()
            self.out_channels = out_channels
            self.hidden_channels = hidden_channels
            self.filter_channels = filter_channels
            self.n_heads = n_heads
            self.n_layers = n_layers
            self.kernel_size = kernel_size
            self.p_dropout = p_dropout
            self.emb_phone = nn.Linear(HUBERT_FEATURE_DIM, hidden_channels)
            self.emb_pitch = nn.Embedding(256, hidden_channels)
            self.lrelu = nn.LeakyReLU(0.1)
            self.encoder = Encoder(hidden_channels, filter_channels, n_heads,
                                   n_layers, kernel_size, p_dropout)
            self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
            self.gin_channels = gin_channels

        def forward(self, phone, pitch, lengths=None):
            # phone: (B, 768, T) -> (B, T, 768); pitch: (B, T) int64.
            x = self.emb_phone(phone.transpose(1, 2)) + self.emb_pitch(pitch)
            x = x * math.sqrt(self.hidden_channels)
            x = self.lrelu(x)
            x = x.transpose(1, 2)  # (B, hidden, T)
            x_mask = torch.ones(x.size(0), 1, x.size(2), device=x.device)
            x = self.encoder(x * x_mask, x_mask)
            stats = self.proj(x) * x_mask
            m, logs = torch.split(stats, self.out_channels, dim=1)
            return x, m, logs, x_mask

    class ResBlock1(nn.Module):
        def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
            super().__init__()
            self.convs1 = nn.ModuleList()
            self.convs2 = nn.ModuleList()
            for i in range(len(dilation)):
                self.convs1.append(
                    weight_norm(nn.Conv1d(channels, channels, kernel_size,
                                          dilation=dilation[i],
                                          padding=(kernel_size * dilation[i] - dilation[i]) // 2))
                )
                self.convs2.append(
                    weight_norm(nn.Conv1d(channels, channels, kernel_size,
                                          dilation=1, padding=(kernel_size - 1) // 2))
                )

        def forward(self, x, x_mask=None):
            for c1, c2 in zip(self.convs1, self.convs2):
                xt = F.leaky_relu(x, 0.1)
                xt = c1(xt)
                xt = F.leaky_relu(xt, 0.1)
                xt = c2(xt)
                x = xt + x
            if x_mask is not None:
                x = x * x_mask
            return x

    class SineGen(nn.Module):
        def __init__(self, samp_rate, harmonic_num=0, sine_amp=0.1,
                     noise_std=0.003, voiced_threshold=0, flag_for_pulse=False):
            super().__init__()
            self.sine_amp = sine_amp
            self.noise_std = noise_std
            self.harmonic_num = harmonic_num
            self.dim = harmonic_num + 1
            self.sampling_rate = samp_rate
            self.voiced_threshold = voiced_threshold

        def _f02uv(self, f0):
            uv = torch.ones_like(f0)
            return uv * (f0 > self.voiced_threshold)

        def forward(self, f0, upp):
            with torch.no_grad():
                f0 = f0[:, None].transpose(1, 2)
                f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim,
                                     device=f0.device)
                f0_buf[:, :, 0] = f0[:, :, 0]
                for idx in np.arange(self.harmonic_num):
                    f0_buf[:, :, idx + 1] = f0_buf[:, :, 0] * (idx + 2)
                rad_values = (f0_buf / self.sampling_rate) % 1
                rand_ini = torch.rand(f0_buf.shape[0], f0_buf.shape[2],
                                      device=f0_buf.device)
                rand_ini[:, 0] = 0
                rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
                tmp_over_one = torch.cumsum(rad_values, 1)
                tmp_over_one *= upp
                tmp_over_one = F.interpolate(
                    tmp_over_one.transpose(2, 1), scale_factor=upp,
                    mode='linear', align_corners=True).transpose(2, 1)
                rad_values = F.interpolate(
                    rad_values.transpose(2, 1), scale_factor=upp,
                    mode='nearest').transpose(2, 1)
                tmp_over_one %= 1
                tmp_over_one_idx = (tmp_over_one[:, 1:, :]
                                    - tmp_over_one[:, :-1, :]) < 0
                cumsum_shift = torch.zeros_like(rad_values)
                cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0
                sine_waves = torch.sin(
                    torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * np.pi)
                sine_waves = sine_waves * self.sine_amp
                uv = self._f02uv(f0)
                uv = F.interpolate(uv.transpose(2, 1), scale_factor=upp,
                                   mode='nearest').transpose(2, 1)
                noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
                noise = noise_amp * torch.randn_like(sine_waves)
                sine_waves = sine_waves * uv + noise
            return sine_waves, uv, noise

    class SourceModuleHnNSF(nn.Module):
        def __init__(self, sampling_rate, harmonic_num=0, sine_amp=0.1,
                     add_noise_std=0.003, voiced_threshod=0, is_half=False):
            super().__init__()
            self.sine_amp = sine_amp
            self.noise_std = add_noise_std
            self.is_half = is_half
            self.l_sin_gen = SineGen(sampling_rate, harmonic_num, sine_amp,
                                     add_noise_std, voiced_threshod)
            self.l_linear = nn.Linear(harmonic_num + 1, 1)
            self.l_tanh = nn.Tanh()

        def forward(self, x, upp=None):
            sine_wavs, uv, _ = self.l_sin_gen(x, upp)
            if self.is_half:
                sine_wavs = sine_wavs.half()
            sine_merge = self.l_tanh(self.l_linear(sine_wavs))
            return sine_merge, None, None

    class GeneratorNSF(nn.Module):
        def __init__(self, initial_channel, resblock_kernel_sizes,
                     resblock_dilation_sizes, upsample_rates, upsample_kernel_sizes,
                     gin_channels, sr, is_half=False):
            super().__init__()
            self.num_kernels = len(resblock_kernel_sizes)
            self.num_upsamples = len(upsample_rates)
            self.conv_pre = nn.Conv1d(initial_channel, _UPSAMPLE_INITIAL, 7, 1, 3)
            self.ups = nn.ModuleList()
            self.resblocks = nn.ModuleList()
            for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
                self.ups.append(
                    weight_norm(nn.ConvTranspose1d(
                        _UPSAMPLE_INITIAL // (2 ** i),
                        _UPSAMPLE_INITIAL // (2 ** (i + 1)),
                        k, u, padding=(k - u) // 2))
                )
            for i in range(len(self.ups)):
                ch = _UPSAMPLE_INITIAL // (2 ** (i + 1))
                for j, (k, d) in enumerate(zip(resblock_kernel_sizes,
                                               resblock_dilation_sizes)):
                    self.resblocks.append(ResBlock1(ch, k, d))
            self.conv_post = nn.Conv1d(ch, 1, 7, 1, 3, bias=False)
            self.upsample_rates = upsample_rates
            self.cond = nn.Conv1d(gin_channels, _UPSAMPLE_INITIAL, 1)
            self.m_source = SourceModuleHnNSF(sampling_rate=sr,
                                              harmonic_num=0)
            # RVC v2 re-injects the source excitation via strided noise_convs.
            self.noise_convs = nn.ModuleList()
            for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
                c_cur = _UPSAMPLE_INITIAL // (2 ** (i + 1))
                if i + 1 < len(upsample_rates):
                    stride_f0 = int(np.prod(upsample_rates[i + 1:]))
                    self.noise_convs.append(
                        nn.Conv1d(1, c_cur, kernel_size=stride_f0 * 2,
                                  stride=stride_f0, padding=stride_f0 // 2)
                    )
                else:
                    self.noise_convs.append(nn.Conv1d(1, c_cur, kernel_size=1))
            self.upp = int(np.prod(upsample_rates))

        def forward(self, x, f0, g=None):
            har_source, _, _ = self.m_source(f0, self.upp)
            har_source = har_source.transpose(1, 2)
            x = self.conv_pre(x)
            if g is not None:
                x = x + self.cond(g)
            for i in range(len(self.ups)):
                x = F.leaky_relu(x, 0.1)
                x = self.ups[i](x)
                x = x + self.noise_convs[i](har_source)
                xs = None
                for j in range(self.num_kernels):
                    res = self.resblocks[i * self.num_kernels + j](x)
                    xs = res if xs is None else xs + res
                x = xs / self.num_kernels
            x = F.leaky_relu(x)
            x = self.conv_post(x)
            x = torch.tanh(x)
            return x

    class ResidualCouplingLayer(nn.Module):
        def __init__(self, channels, hidden_channels, kernel_size, dilation_rate,
                     n_layers, gin_channels=0, mean_only=True):
            super().__init__()
            assert channels % 2 == 0, 'channels should be divisible by 2'
            self.channels = channels
            self.hidden_channels = hidden_channels
            self.kernel_size = kernel_size
            self.dilation_rate = dilation_rate
            self.n_layers = n_layers
            self.half_channels = channels // 2
            self.mean_only = mean_only
            self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
            self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers,
                          gin_channels)
            self.post = nn.Conv1d(hidden_channels,
                                  self.half_channels * (2 - mean_only), 1)

        def forward(self, x, x_mask, g=None, reverse=False):
            x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
            h = self.pre(x0) * x_mask
            h = self.enc(h, x_mask, g=g)
            stats = self.post(h) * x_mask
            if not self.mean_only:
                m, logs = torch.split(stats, [self.half_channels] * 2, 1)
            else:
                m = stats
                logs = torch.zeros_like(m)
            if not reverse:
                x1 = m + x1 * torch.exp(logs) * x_mask
                x = torch.cat([x0, x1], 1)
                return x, None
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x

    class ResidualCouplingBlock(nn.Module):
        def __init__(self, channels, hidden_channels, kernel_size, dilation_rate,
                     n_layers, n_flows=4, gin_channels=0):
            super().__init__()
            self.flows = nn.ModuleList()
            for i in range(n_flows):
                self.flows.append(
                    ResidualCouplingLayer(channels, hidden_channels, kernel_size,
                                          dilation_rate, n_layers, gin_channels)
                )
                self.flows.append(Flip())

        def forward(self, x, x_mask, g=None, reverse=False):
            if not reverse:
                for flow in self.flows:
                    x, _ = flow(x, x_mask, g=g, reverse=reverse)
            else:
                for flow in reversed(self.flows):
                    x = flow(x, x_mask, g=g, reverse=reverse)
            return x

    class Flip(nn.Module):
        def forward(self, x, *args, reverse=False, **kwargs):
            x = torch.flip(x, [1])
            if not reverse:
                return x, None
            return x

    class SynthesizerTrnMs768NSFsid(nn.Module):
        def __init__(self):
            super().__init__()
            self.spk_embed_dim = _SPEAKER_EMBED_DIM
            self.emb_g = nn.Embedding(_N_SPEAKERS, _SPEAKER_EMBED_DIM)
            self.enc_p = TextEncoder(
                _INTER_CHANNELS, _HIDDEN_CHANNELS, _FILTER_CHANNELS,
                _N_HEADS, 6, 3, 0.0, _GIN_CHANNELS, self.spk_embed_dim
            )
            self.dec = GeneratorNSF(
                _INTER_CHANNELS, _RESBLOCK_KERNELS, _RESBLOCK_DILATIONS,
                _UPSAMPLE_RATES, _UPSAMPLE_KERNELS, _GIN_CHANNELS,
                RVC_SAMPLE_RATE
            )
            self.flow = ResidualCouplingBlock(
                _INTER_CHANNELS, _HIDDEN_CHANNELS, 5, 1, 3, 4, _GIN_CHANNELS
            )

        def infer(self, phone, phone_lengths, pitch, nsff0, sid, rate=None):
            g = self.emb_g(sid).unsqueeze(-1)
            x, m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
            z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
            if rate:
                head = int(z_p.shape[2] * rate)
                z_p = z_p[:, :, -head:]
                x_mask = x_mask[:, :, -head:]
                nsff0 = nsff0[:, -head:]
            z = self.flow(z_p, x_mask, g=g, reverse=True)
            o = self.dec(z * x_mask, nsff0, g=g)
            return o, x_mask, (z, z_p, m_p, logs_p)

    return SynthesizerTrnMs768NSFsid


@dataclass
class RVCConfig:
    model_path: Path
    index_path: Path | None = None
    device: str = 'cuda:0'
    index_rate: float = 0.5
    f0_method: str = 'rmvpe'
    speaker_id: int = 0


class RVCVoiceConverter:
    """Convert a 24 kHz wav to the target voice (RVC v2, 40 kHz synthesis)."""

    def __init__(
        self,
        model_path: Path,
        index_path: Path | None = None,
        device: str | None = None,
        index_rate: float = 0.5,
        f0_method: str = 'rmvpe',
        speaker_id: int = 0,
    ):
        import torch

        self.model_path = Path(model_path)
        self.index_path = Path(index_path) if index_path else None
        self.index_rate = float(index_rate)
        self.f0_method = f0_method
        self.speaker_id = speaker_id
        self.device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.index = None
        self._big_npy = None
        self._net = None
        self._contentvec = None
        self._f0_model = None
        self._load_index()

    # -- lazy loading ------------------------------------------------------ #
    def _load_index(self) -> None:
        if not self.index_path or not Path(self.index_path).exists():
            self.index = None
            self._big_npy = None
            return
        import faiss
        import numpy as np

        self.index = faiss.read_index(str(self.index_path))
        self._big_npy = self.index.reconstruct_n(0, self.index.ntotal)
        logger.info(f'🔎 [RVC] Đã nạp FAISS index: {self.index_path} '
                    f'({self.index.ntotal} vectors)')

    def _load_net(self):
        if self._net is not None:
            return self._net
        import torch

        cls = _build_models_classes()
        net = cls()
        state = torch.load(str(self.model_path), map_location='cpu',
                           weights_only=False)
        state = state.get('model', state)
        missing, unexpected = net.load_state_dict(state, strict=False)
        missing = [m for m in missing if not m.startswith('enc_q.')]
        if missing:
            logger.warning(f'⚠️ [RVC] Thiếu tham số khi nạp model: {missing[:8]}...')
        net.eval().to(self.device)
        self._net = net
        logger.info(f'🎛️ [RVC] Đã nạp generator: {self.model_path} -> {self.device}')
        return net

    def _load_contentvec(self):
        if self._contentvec is not None:
            return self._contentvec
        import torch
        from transformers import HubertModel

        model = HubertModel.from_pretrained(CONTENTVEC_REPO)
        model.eval().to(self.device)
        self._contentvec = model
        logger.info('🧬 [RVC] Đã nạp ContentVec (768-dim).')
        return model

    def _load_f0_model(self):
        if self._f0_model is not None:
            return self._f0_model
        # RMVPE is RVC's preferred extractor, but its DeepUnet+GRU topology is a
        # large reconstruction. librosa.pyin is accurate, already in the venv,
        # and is the documented fallback the plan allows.
        self._f0_model = None
        return None

    # -- stages ------------------------------------------------------------ #
    def _extract_features(self, audio, sr):
        import numpy as np
        import torch

        model = self._load_contentvec()
        if sr != 16000:
            import librosa

            audio = librosa.resample(np.asarray(audio, dtype=np.float32),
                                     orig_sr=sr, target_sr=16000)
        with torch.no_grad():
            wav = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
            wav = wav.to(self.device)
            out = model(wav).last_hidden_state
        feats = out.transpose(1, 2).contiguous()
        if self.index is not None and self.index_rate > 0:
            feats = self._retrieve(feats)
        return feats

    def _retrieve(self, feats):
        """RVC's feature retrieval: k=8 neighbours, inverse-square-distance blend."""
        import numpy as np
        import torch

        npy = feats.squeeze(0).transpose(0, 1).cpu().numpy().astype(np.float32)
        score, ix = self.index.search(npy, k=8)
        weight = np.square(1 / (score + 1e-9))
        weight /= weight.sum(axis=1, keepdims=True)
        big = self._big_npy
        blended = np.sum(big[ix] * weight[:, :, None], axis=1)
        o_npy = (self.index_rate * blended
                 + (1 - self.index_rate) * npy).astype(np.float32)
        return torch.from_numpy(o_npy).transpose(0, 1).unsqueeze(0).to(self.device)

    def _extract_f0(self, audio, sr):
        """F0 (Hz) at RVC's 100 Hz frame grid over the 16 kHz signal."""
        import numpy as np
        import librosa

        audio = np.asarray(audio, dtype=np.float32)
        if sr != RVC_FEATURE_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=RVC_FEATURE_SR)
        hop = RVC_WINDOW * 2  # 320 samples @16k = 20 ms == ContentVec frame rate
        f0, _, _ = librosa.pyin(
            audio,
            fmin=RVC_F0_MIN,
            fmax=RVC_F0_MAX,
            sr=RVC_FEATURE_SR,
            frame_length=2048,
            hop_length=hop,
        )
        f0 = np.nan_to_num(np.asarray(f0, dtype=np.float32))
        return self._coarse_f0(f0), f0

    @staticmethod
    def _coarse_f0(f0):
        """RVC's f0_to_coarse: mel-warp into integer bins 1..255 (0 -> 1)."""
        import numpy as np

        f0 = np.asarray(f0, dtype=np.float32)
        f0_mel_min = 1127.0 * np.log(1 + RVC_F0_MIN / 700.0)
        f0_mel_max = 1127.0 * np.log(1 + RVC_F0_MAX / 700.0)
        f0_mel = 1127.0 * np.log(1 + f0 / 700.0)
        pos = f0_mel > 0
        f0_mel[pos] = (f0_mel[pos] - f0_mel_min) * 254.0 / (
            f0_mel_max - f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        return np.rint(f0_mel).astype(np.int64)

    def _run_net(self, feats, pitch, pitchf, sid):
        import numpy as np
        import torch

        net = self._load_net()
        phone = feats.to(self.device)
        p_len = min(phone.shape[2], pitch.shape[0], pitchf.shape[0])
        phone = phone[:, :, :p_len]
        pitch_t = torch.from_numpy(pitch[:p_len]).unsqueeze(0).long().to(self.device)
        pitchf_t = torch.from_numpy(pitchf[:p_len]).unsqueeze(0).float().to(self.device)
        sid_t = torch.LongTensor([sid]).to(self.device)
        with torch.no_grad():
            o, _, _ = net.infer(phone, None, pitch_t, pitchf_t, sid_t)
        return o.squeeze().cpu().numpy().astype(np.float32)

    # -- public API -------------------------------------------------------- #
    def convert_file(self, in_wav: Path, out_wav: Path) -> int:
        """Convert `in_wav` to `out_wav` (mono PCM_16 @ 40 kHz).

        Returns the output duration in milliseconds.
        """
        import numpy as np
        import soundfile as sf

        in_wav = Path(in_wav)
        out_wav = Path(out_wav)
        out_wav.parent.mkdir(parents=True, exist_ok=True)

        audio, sr = sf.read(str(in_wav), dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        feats = self._extract_features(audio, sr)
        pitch, pitchf = self._extract_f0(audio, sr)
        out = self._run_net(feats, pitch, pitchf, self.speaker_id)
        out = np.asarray(out, dtype=np.float32).reshape(-1)

        pcm16 = np.clip(out, -1.0, 1.0)
        pcm16 = (pcm16 * 32767.0).astype('<i2')
        with wave.open(str(out_wav), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(RVC_SAMPLE_RATE)
            wf.writeframes(pcm16.tobytes())
        return int(round(len(out) / RVC_SAMPLE_RATE * 1000))


def convert_wav_to_24k(
    in_wav: Path,
    out_wav: Path,
    converter: RVCVoiceConverter,
) -> int:
    """Run RVC then ffmpeg-resample the 40 kHz result back to 24 kHz PCM_16."""
    import subprocess
    import tempfile

    in_wav = Path(in_wav)
    out_wav = Path(out_wav)
    with tempfile.TemporaryDirectory() as tmpd:
        raw40 = Path(tmpd) / 'rvc40.wav'
        converter.convert_file(in_wav, raw40)
        ffmpeg_bin = _ffmpeg_bin()
        subprocess.run(
            [
                ffmpeg_bin, '-y', '-v', 'error',
                '-i', str(raw40),
                '-ac', '1', '-ar', str(SAMPLE_RATE), '-sample_fmt', 's16',
                str(out_wav),
            ],
            check=True,
        )
    with wave.open(str(out_wav), 'rb') as wf:
        return int(round(wf.getnframes() / wf.getframerate() * 1000))


def _ffmpeg_bin() -> str:
    import shutil

    venv_ffmpeg = Path(__file__).resolve().parent.parent / '.venv' / 'bin' / 'ffmpeg'
    if venv_ffmpeg.exists():
        return str(venv_ffmpeg)
    return shutil.which('ffmpeg') or 'ffmpeg'
