"""
Viterbox - Vietnamese Text-to-Speech
Based on Chatterbox architecture, fine-tuned for Vietnamese.
"""
import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Union, List

from huggingface_hub import snapshot_download
from safetensors.torch import load_file as load_safetensors

from .models.t3 import T3, T3Config
from .models.t3.modules.cond_enc import T3Cond
from .models.s3gen import S3Gen, S3GEN_SR
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.voice_encoder import VoiceEncoder
from .models.tokenizers import MTLTokenizer

try:
    from soe_vinorm import SoeNormalizer
    _normalizer = SoeNormalizer()
    HAS_VINORM = True
except ImportError:
    HAS_VINORM = False
    _normalizer = None


REPO_ID = "dolly-vn/viterbox"
WAVS_DIR = Path("wavs")


# Global VAD model
_VAD_MODEL = None
_VAD_UTILS = None


def get_vad_model():
    """Load Silero VAD model (singleton)"""
    global _VAD_MODEL, _VAD_UTILS
    if _VAD_MODEL is None:
        try:
            # Load from torch hub - will be cached
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                trust_repo=True,
                verbose=False
            )
            _VAD_MODEL = model
            _VAD_UTILS = utils
        except Exception as e:
            print(f"⚠️ Could not load Silero VAD: {e}")
            return None, None
    return _VAD_MODEL, _VAD_UTILS


def get_random_voice() -> Optional[Path]:
    """Get a random voice file from wavs folder"""
    if WAVS_DIR.exists():
        voices = list(WAVS_DIR.glob("*.wav"))
        if voices:
            import random
            return random.choice(voices)
    return None


def punc_norm(text: str) -> str:
    """
    Quick cleanup func for punctuation from LLMs or
    containing chars not seen often in the dataset
    """
    if len(text) == 0:
        return "You need to add some text for me to talk."

    # Capitalise first letter
    if len(text) > 0 and text[0].islower():
        text = text[0].upper() + text[1:]

    # Remove multiple space chars
    text = " ".join(text.split())

    # Replace uncommon/llm punc
    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ('"', '"'),
        ("'", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # Add full stop if no ending punc
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ",", "、", "，", "。", "？", "！"}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text


def normalize_text(text: str, language: str = "vi") -> str:
    """Normalize Vietnamese text (numbers, abbreviations, etc.)"""
    if language == "vi" and HAS_VINORM and _normalizer is not None:
        try:
            return _normalizer.normalize(text)
        except Exception:
            return text
    return text


def _split_text_to_sentences(text: str, max_words: int = 30) -> List[str]:
    """Split text into sentences by punctuation marks.

    Only sentences of at least 30 words that contain a comma are further split
    on ``,`` / ``;`` / ``–—``; each emitted clause preserves the original
    punctuation (a comma clause ends with a real ``,``, never a forced ``.``).
    Splitting always happens *at* a punctuation boundary — never in the middle
    of a word — so T3 is never handed an unpunctuated fragment to break.
    """
    # Split by . ? ! and keep the delimiter
    pattern = r'([.?!]+)'
    parts = re.split(pattern, text)

    sentences = []
    current = ""
    for i, part in enumerate(parts):
        if re.match(pattern, part):
            # This is punctuation, append to current sentence
            current += part
            if current.strip():
                sentences.append(current.strip())
            current = ""
        else:
            current = part

    # Don't forget remaining text without ending punctuation
    if current.strip():
        sentences.append(current.strip())

    sentences = [s for s in sentences if s.strip()]

    # Second pass: break long comma-clause sentences into short clauses.
    chunked: List[str] = []
    for sentence in sentences:
        if len(sentence.split()) < 30 or ',' not in sentence:
            chunked.append(sentence)
            continue
        # Split keeping the separating punctuation on the left piece.
        pieces = re.split(r'(?<=[,;–—])\s+', sentence)
        current_clause = ""
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            if current_clause and len(current_clause.split()) + len(piece.split()) > max_words:
                chunked.append(current_clause)
                current_clause = piece
            elif not current_clause:
                current_clause = piece
            else:
                current_clause = f'{current_clause} {piece}'
        if current_clause:
            chunked.append(current_clause)

    return [c for c in (chunked or sentences) if c.strip()]


def trim_silence(audio: np.ndarray, sr: int, top_db: int = 30) -> np.ndarray:
    """Legacy trim silence (energy based)."""
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed


def vad_trim(audio: np.ndarray, sr: int, margin_s: float = 0.01) -> np.ndarray:
    """
    Trim audio using Silero VAD to strictly keep only speech.
    
    Args:
        audio: Audio array (numpy)
        sr: Sample rate
        margin_s: Margin to keep after speech ends (seconds)
    """
    if len(audio) == 0:
        return audio
        
    model, utils = get_vad_model()
    if model is None:
        return trim_silence(audio, sr, top_db=20)
        
    (get_speech_timestamps, _, read_audio, *_) = utils
    
    # Prepare audio for VAD (must be float32)
    wav = torch.tensor(audio, dtype=torch.float32)
    
    # If sampling rate is not 8k or 16k, we might need resample for VAD? 
    # Silero supports 8000 or 16000 directly usually, but newer versions handle others.
    # We will trust utils to handle or just pass as is (Silero supports 16k best).
    
    # Actually Silero expects simple tensor. Let's try direct.
    # Note: Silero often works best at 16k.
    
    try:
        # Get speech timestamps
        # VAD typically expects 16000 sr. Let's resample strictly for detection if needed
        # but let's try direct first. If sr is 24000, silero might warn.
        # Safe bet: resample local copy for detection
        
        vad_sr = 16000
        if sr != vad_sr:
            # Quick resample for detection only
            wav_16k = librosa.resample(audio, orig_sr=sr, target_sr=vad_sr)
            wav_tensor = torch.tensor(wav_16k, dtype=torch.float32)
        else:
            wav_tensor = wav
            
        # Use VAD parameters
        timestamps = get_speech_timestamps(
            wav_tensor, 
            model, 
            sampling_rate=vad_sr, 
            threshold=0.35,  # Relax threshold as we fixed the root cause
            min_speech_duration_ms=250, 
            min_silence_duration_ms=100
        )
        
        if not timestamps:
            # No speech detected? Fallback to mild energy trim or return as is?
            # Sometimes VAD misses breathy endings. Let's fallback to energy trim
            return trim_silence(audio, sr, top_db=25)
            
        # Get end of last speech chunk
        last_end_sample_16k = timestamps[-1]['end']
        
        # Convert back to original sample rate
        last_end_sample = int(last_end_sample_16k * (sr / vad_sr))
        
        # Add margin
        margin_samples = int(margin_s * sr)
        cut_point = last_end_sample + margin_samples
        
        # Don't cut beyond length
        cut_point = min(cut_point, len(audio))
        
        # Trim
        return audio[:cut_point]
        
    except Exception as e:
        print(f"⚠️ VAD Error: {e}")
        return trim_silence(audio, sr, top_db=20)


def apply_fade_out(audio: np.ndarray, sr: int, fade_duration: float = 0.01) -> np.ndarray:
    """
    Apply smooth fade-out to prevent click artifacts at the end of audio.
    
    Args:
        audio: Audio array
        sr: Sample rate
        fade_duration: Fade duration in seconds (default 10ms)
    
    Returns:
        Audio with fade-out applied
    """
    if len(audio) == 0:
        return audio
    
    fade_samples = int(fade_duration * sr)
    fade_samples = min(fade_samples, len(audio))  # Don't fade more than audio length
    
    if fade_samples <= 0:
        return audio
    
    # Create fade-out curve (linear)
    fade_curve = np.linspace(1.0, 0.0, fade_samples)
    
    # Apply fade to end of audio
    audio_copy = audio.copy()
    audio_copy[-fade_samples:] = audio_copy[-fade_samples:] * fade_curve
    
    return audio_copy


def apply_fade_in(audio: np.ndarray, sr: int, fade_duration: float = 0.005) -> np.ndarray:
    """
    Apply smooth fade-in to prevent click artifacts at the start of audio.
    
    Args:
        audio: Audio array
        sr: Sample rate
        fade_duration: Fade duration in seconds (default 5ms)
    
    Returns:
        Audio with fade-in applied
    """
    if len(audio) == 0:
        return audio
    
    fade_samples = int(fade_duration * sr)
    fade_samples = min(fade_samples, len(audio))
    
    if fade_samples <= 0:
        return audio
    
    # Create fade-in curve (linear)
    fade_curve = np.linspace(0.0, 1.0, fade_samples)
    
    # Apply fade to start of audio
    audio_copy = audio.copy()
    audio_copy[:fade_samples] = audio_copy[:fade_samples] * fade_curve
    
    return audio_copy


def _frame_rms_zcr(
    audio: np.ndarray,
    sr: int,
    frame_ms: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Split ``audio`` into ``frame_ms`` frames and return per-frame (RMS, ZCR).

    RMS is the root-mean-square amplitude, ZCR the zero-crossing rate
    (crossings / (frame_len - 1)). Frames are non-overlapping (hop == frame).
    """
    frame_len = max(1, int(sr * frame_ms / 1000.0))
    n_frames = len(audio) // frame_len
    if n_frames == 0:
        rms = np.array([float(np.sqrt(np.mean(audio ** 2)))]) if len(audio) else np.array([])
        zcr = np.array([0.0])
        return rms, zcr

    frames = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    signs = frames >= 0
    crossings = np.count_nonzero(signs[:, 1:] != signs[:, :-1], axis=1)
    zcr = crossings.astype(np.float64) / max(1, frame_len - 1)
    return rms, zcr


def tail_cleanup_clean(
    audio: np.ndarray,
    sr: int,
    frame_ms: float = 10.0,
    body_rms: float = 0.02,
    valley_rms: float = 0.0055,
    valley_frames: int = 3,
    search_tail_s: float = 0.6,
    zcr_noise: float = 0.12,
    zcr_rms: float = 0.04,
    fade_in_ms: float = 10.0,
    fade_out_ms: float = 25.0,
    head_keep_ms: float = 18.0,
) -> np.ndarray:
    """Cut the buzzing tail artifact *and* the redundant head silence.

    S3Gen's final buffers contain, after the real speech has decayed to a
    silence valley (RMS < ``valley_rms``), a short high-ZCR noise island
    (RMS up to ~0.036, ZCR 0.15-0.36) that is heard as a "rè rè"/buzz. A
    right-to-left trim cannot remove it because the noise island itself stops
    the trim. This routine instead scans the last ``search_tail_s`` seconds
    left-to-right: once the speech body (RMS > ``body_rms``) has been seen and
    the signal then drops into a valley lasting >= ``valley_frames`` frames
    (30 ms), everything from the *start of that valley* onward is discarded —
    the valley **and** every trailing noise island after it.

    Independently, the leading silence S3Gen pads onto every buffer is trimmed
    down to at most ``head_keep_ms`` before the first speech frame. Without
    this, concatenating clauses accumulates ~150-200 ms of dead air on top of
    the programmed pause. A Hann fade-in (``fade_in_ms``) then ramps the kept
    head up cleanly.

    Any residual weak/high-ZCR frames at the very end are dropped too, then a
    Hann fade-out (``fade_out_ms``) drives the amplitude to exactly 0.0 so the
    buffer joins a pause without a click.

    Returns a new array (never longer than the input).
    """
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    n = len(audio)
    if n == 0:
        return audio

    frame_len = max(1, int(sr * frame_ms / 1000.0))
    rms, zcr = _frame_rms_zcr(audio, sr, frame_ms)
    n_frames = len(rms)
    if n_frames == 0:
        return _apply_hann_fades(audio, sr, fade_in_ms, fade_out_ms)

    # Only inspect the trailing window where the artifact lives.
    tail_frames = min(n_frames, max(1, int(search_tail_s * 1000.0 / frame_ms)))
    start_frame = n_frames - tail_frames

    cut_frame = n_frames  # exclusive end frame kept
    seen_body = False
    valley_run = 0
    for f in range(start_frame, n_frames):
        r = rms[f]
        if r > body_rms:
            seen_body = True
            valley_run = 0
            continue
        if seen_body and r < valley_rms:
            if valley_run == 0:
                # Mark the start of the valley; if it lasts long enough we cut here.
                valley_start = f
            valley_run += 1
            if valley_run >= valley_frames:
                cut_frame = valley_start
                break
        elif r >= valley_rms:
            # Anything at/above valley level (including a noise island) breaks
            # a forming valley only if it is a real speech body; weak high-ZCR
            # islands must not reset the valley counter.
            if r > body_rms:
                valley_run = 0
            else:
                valley_run = max(0, valley_run)

    # Trim trailing weak/high-ZCR noise frames that survived the valley cut.
    last = cut_frame
    while last > 1:
        r = rms[last - 1]
        z = zcr[last - 1]
        if r < zcr_rms and z > zcr_noise:
            last -= 1
        else:
            break
    cut_frame = last

    keep_frames = min(cut_frame, n_frames)
    end = min(n, keep_frames * frame_len)
    cleaned = audio[:end]
    if len(cleaned) == 0:
        return audio[:0]

    # Trim redundant leading silence S3Gen pads onto every buffer: keep at most
    # ``head_keep_ms`` before the first speech frame so concatenated clauses do
    # not accumulate dead air on top of the programmed pause.
    head_frames = min(n_frames, max(1, int(head_keep_ms * 1000.0 / frame_ms)))
    first_speech = None
    scan_frames = min(n_frames, max(head_frames, int(0.5 * 1000.0 / frame_ms)))
    for f in range(scan_frames):
        if rms[f] > body_rms:
            first_speech = f
            break
    if first_speech is not None and first_speech > 0:
        head_samples = max(0, first_speech - head_frames) * frame_len
        cleaned = cleaned[head_samples:]
    if len(cleaned) == 0:
        return audio[:0]

    return _apply_hann_fades(cleaned, sr, fade_in_ms, fade_out_ms)


def _apply_hann_fades(
    audio: np.ndarray, sr: int, fade_in_ms: float, fade_out_ms: float
) -> np.ndarray:
    """Apply a raised-cosine fade-in/out so the buffer starts and ends at 0.0.

    The fade-out uses a *squared* raised-cosine: it rolls off faster near the
    end, so the final ~5 ms is already effectively silent (guaranteeing the
    clause joins the following pause without any residual buzz).
    """
    out = audio.copy()
    n = len(out)
    if n == 0:
        return out

    fin = min(int(sr * fade_in_ms / 1000.0), n)
    if fin > 1:
        out[:fin] *= np.hanning(fin * 2)[:fin]

    fout = min(int(sr * fade_out_ms / 1000.0), n)
    if fout > 1:
        curve = np.hanning(fout * 2)[fout:]  # 1 -> 0 raised cosine
        out[-fout:] *= curve ** 2  # faster roll-off near the end

    # Hard-zero the last 1 ms to guarantee a dead-silent edge into the pause.
    z = min(int(sr * 0.001), n)
    if z:
        out[-z:] = 0.0
    out[-1] = 0.0
    return out


def crossfade_concat(audios: List[np.ndarray], sr: int, fade_ms: int = 50, pause_ms: int = 500) -> np.ndarray:
    """
    Concatenate audio segments with crossfading and optional pause between sentences.
    
    Args:
        audios: List of audio arrays
        sr: Sample rate
        fade_ms: Crossfade duration in milliseconds
        pause_ms: Pause duration between sentences in milliseconds
    """
    if not audios:
        return np.array([])
    if len(audios) == 1:
        return audios[0]
    
    fade_samples = int(sr * fade_ms / 1000)
    pause_samples = int(sr * pause_ms / 1000)
    
    # Build result
    result = audios[0].copy()
    
    for i in range(1, len(audios)):
        next_audio = audios[i]
        
        # Add pause between sentences
        if pause_samples > 0:
            silence = np.zeros(pause_samples, dtype=result.dtype)
            result = np.concatenate([result, silence])
        
        if len(result) < fade_samples or len(next_audio) < fade_samples:
            # Too short for crossfade, just concatenate
            result = np.concatenate([result, next_audio])
            continue
        
        # Create fade curves
        fade_out = np.linspace(1.0, 0.0, fade_samples)
        fade_in = np.linspace(0.0, 1.0, fade_samples)
        
        # Apply crossfade
        result_end = result[-fade_samples:] * fade_out
        next_start = next_audio[:fade_samples] * fade_in
        crossfaded = result_end + next_start
        
        # Combine
        result = np.concatenate([
            result[:-fade_samples],
            crossfaded,
            next_audio[fade_samples:]
        ])
    
    return result


@dataclass
class TTSConds:
    """Conditioning tensors for TTS generation"""
    t3: Union['T3Cond', dict]  # T3 conditioning (T3Cond object or dict)
    s3: dict  # S3Gen conditioning dict
    ref_wav: Optional[torch.Tensor] = None
    
    def save(self, path):
        def to_cpu(x):
            if isinstance(x, torch.Tensor):
                return x.cpu()
            elif isinstance(x, dict):
                return {k: to_cpu(v) for k, v in x.items()}
            elif hasattr(x, '__dict__'):
                return {k: to_cpu(v) for k, v in vars(x).items()}
            return x
        
        torch.save({
            't3': to_cpu(self.t3),
            'gen': to_cpu(self.s3),
        }, path)
    
    @classmethod
    def load(cls, path, device):
        def to_device(x, dev):
            if isinstance(x, torch.Tensor):
                return x.to(dev)
            elif isinstance(x, dict):
                return {k: to_device(v, dev) for k, v in x.items()}
            return x
        
        data = torch.load(path, map_location='cpu', weights_only=False)
        
        # Handle both old format (t3, s3) and new format (t3, gen)
        t3_data = data.get('t3', {})
        s3_data = data.get('gen', data.get('s3', {}))
        ref_wav = data.get('ref_wav', None)
        
        # Convert t3_data dict to T3Cond object
        if isinstance(t3_data, dict) and 'speaker_emb' in t3_data:
            t3_cond = T3Cond(
                speaker_emb=to_device(t3_data['speaker_emb'], device),
                cond_prompt_speech_tokens=to_device(t3_data.get('cond_prompt_speech_tokens'), device),
                cond_prompt_speech_emb=to_device(t3_data.get('cond_prompt_speech_emb'), device) if t3_data.get('cond_prompt_speech_emb') is not None else None,
                clap_emb=to_device(t3_data.get('clap_emb'), device) if t3_data.get('clap_emb') is not None else None,
                emotion_adv=to_device(t3_data.get('emotion_adv'), device) if t3_data.get('emotion_adv') is not None else None,
            )
        else:
            t3_cond = to_device(t3_data, device)
        
        return cls(
            t3=t3_cond,
            s3=to_device(s3_data, device),
            ref_wav=to_device(ref_wav, device) if ref_wav is not None else None,
        )


class Viterbox:
    """
    Vietnamese Text-to-Speech model.
    
    Example:
        >>> tts = Viterbox.from_pretrained("cuda")
        >>> audio = tts.generate("Xin chào!")
        >>> tts.save_audio(audio, "output.wav")
    """
    
    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: MTLTokenizer,
        device: str = "cuda",
    ):
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.sr = 24000  # Output sample rate
        self.conds: Optional[TTSConds] = None
        
    @classmethod
    def from_pretrained(cls, device: str = "cuda") -> 'Viterbox':
        """Load model from HuggingFace Hub to local pretrained directory"""
        # Tải về thư mục pretrained/ cục bộ trong dự án
        local_pretrained_dir = Path(__file__).parent.parent / "pretrained"
        local_pretrained_dir.mkdir(parents=True, exist_ok=True)
        
        ckpt_dir = Path(
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="model",
                revision="main",
                allow_patterns=[
                    "ve.pt",
                    "t3_ml24ls_v2.safetensors",
                    "s3gen.pt",
                    "tokenizer_vi_expanded.json",
                    "conds.pt",
                ],
                token=os.getenv("HF_TOKEN"),
            )
        )
        return cls.from_local(ckpt_dir, device)
    
    @classmethod
    def from_local(cls, ckpt_dir: Union[str, Path], device: str = "cuda") -> 'Viterbox':
        """Load model from local directory"""
        ckpt_dir = Path(ckpt_dir)
        
        # Load Voice Encoder
        ve = VoiceEncoder()
        if device == "mps":
            ve.load_state_dict(torch.load(ckpt_dir / "ve.pt", map_location='cpu',weights_only=True))
        else:
            ve.load_state_dict(torch.load(ckpt_dir / "ve.pt", weights_only=True))
        ve.to(device).eval()
        
        # Load T3 model
        t3 = T3(T3Config.multilingual())
        t3_state = load_safetensors(ckpt_dir / "t3_ml24ls_v2.safetensors")
        
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        
        # Resize embeddings if needed
        if "text_emb.weight" in t3_state:
            old_emb = t3_state["text_emb.weight"]
            if old_emb.shape[0] != t3.hp.text_tokens_dict_size:
                new_emb = torch.zeros((t3.hp.text_tokens_dict_size, old_emb.shape[1]), dtype=old_emb.dtype)
                min_rows = min(old_emb.shape[0], new_emb.shape[0])
                new_emb[:min_rows] = old_emb[:min_rows]
                if new_emb.shape[0] > min_rows:
                    nn.init.normal_(new_emb[min_rows:], mean=0.0, std=0.02)
                t3_state["text_emb.weight"] = new_emb
        
        if "text_head.weight" in t3_state:
            old_head = t3_state["text_head.weight"]
            if old_head.shape[0] != t3.hp.text_tokens_dict_size:
                new_head = torch.zeros((t3.hp.text_tokens_dict_size, old_head.shape[1]), dtype=old_head.dtype)
                min_rows = min(old_head.shape[0], new_head.shape[0])
                new_head[:min_rows] = old_head[:min_rows]
                if new_head.shape[0] > min_rows:
                    nn.init.normal_(new_head[min_rows:], mean=0.0, std=0.02)
                t3_state["text_head.weight"] = new_head
        
        t3.load_state_dict(t3_state)
        t3.to(device).eval()
        
        # Load S3Gen
        s3gen = S3Gen()
        if device == "mps":
            s3gen.load_state_dict(torch.load(ckpt_dir / "s3gen.pt", map_location='cpu',weights_only=True))
        else:
            s3gen.load_state_dict(torch.load(ckpt_dir / "s3gen.pt", weights_only=True))
        s3gen.to(device).eval()
        
        # Load tokenizer
        tokenizer = MTLTokenizer(str(ckpt_dir / "tokenizer_vi_expanded.json"))
        
        model = cls(t3, s3gen, ve, tokenizer, device)
        
        # Load default conditioning if exists
        conds_path = ckpt_dir / "conds.pt"
        if conds_path.exists():
            model.conds = TTSConds.load(conds_path, device)
        
        return model
    
    def prepare_conditionals(
        self,
        audio_prompt: Union[str, Path, torch.Tensor],
        exaggeration: float = 0.5,
        prompt_time_stretch: float = 1.0,
    ):
        """
        Prepare conditioning from reference audio.
        
        Args:
            audio_prompt: Path to WAV file or audio tensor
            exaggeration: Expression intensity (0.0 - 2.0)
            prompt_time_stretch: Rate applied to the 16 kHz prompt copy that
                feeds the T3 speech-conditioning tokens. Values < 1 slow the
                prompt's rhythm, so T3 clones the *timbre* of the reference
                while reading at a calmer, clearly-articulated pace. The
                S3Gen/voice-encoder waveforms are left untouched to preserve
                the speaker timbre exactly.
        """
        # Load audio at S3Gen sample rate (24kHz)
        if isinstance(audio_prompt, (str, Path)):
            s3gen_ref_wav, _ = librosa.load(str(audio_prompt), sr=S3GEN_SR, mono=True)
        else:
            s3gen_ref_wav = audio_prompt.cpu().numpy()
            if s3gen_ref_wav.ndim > 1:
                s3gen_ref_wav = s3gen_ref_wav.squeeze()
        
        # Resample to 16kHz for voice encoder and tokenizer
        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)
        
        # Limit conditioning length
        DEC_COND_LEN = S3GEN_SR * 10  # 10 seconds max
        ENC_COND_LEN = S3_SR * 10
        s3gen_ref_wav = s3gen_ref_wav[:DEC_COND_LEN]
        
        # The T3 speech-conditioning prompt (which carries the reading *pace*)
        # gets the tempo change; the S3Gen/voice-encoder copies do not, so the
        # cloned timbre stays bit-for-bit identical to the golden reference.
        prompt_16k = ref_16k_wav
        if prompt_time_stretch and abs(prompt_time_stretch - 1.0) > 1e-3:
            prompt_16k = librosa.effects.time_stretch(ref_16k_wav, rate=float(prompt_time_stretch))
        
        with torch.inference_mode():
            # Get S3Gen conditioning
            s3_cond = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)
            
            # Speech cond prompt tokens for T3
            t3_cond_prompt_tokens = None
            if plen := self.t3.hp.speech_cond_prompt_len:
                s3_tokzr = self.s3gen.tokenizer
                t3_cond_prompt_tokens, _ = s3_tokzr.forward([prompt_16k[:ENC_COND_LEN]], max_len=plen)
                t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)
            
            # Voice-encoder speaker embedding (original 16k prompt: timbre).
            ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
            ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)
            
            # Create T3Cond
            t3_cond = T3Cond(
                speaker_emb=ve_embed,
                cond_prompt_speech_tokens=t3_cond_prompt_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)
        
        self.conds = TTSConds(t3=t3_cond, s3=s3_cond, ref_wav=torch.from_numpy(s3gen_ref_wav).unsqueeze(0))
        return self.conds
    
    def _generate_single(
        self,
        text: str,
        language: str,
        cfg_weight: float,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
    ) -> np.ndarray:
        # Normalize and ensure text ends with punctuation (crucial for T3)
        text = punc_norm(text)
            
        # Tokenize text with language prefix
        text_tokens = self.tokenizer.text_to_tokens(text, language_id=language).to(self.device)
        
        # Duplicate for CFG (classifier-free guidance needs two sequences)
        text_tokens = torch.cat([text_tokens, text_tokens], dim=0)
        
        # Add start and stop tokens
        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        # Automatically detect device type to enable Autocast accordingly
        use_autocast = self.device in ['cuda', 'mps']
        device_type = 'cuda' if self.device == 'cuda' else 'mps'

        with torch.inference_mode(), torch.autocast(device_type=device_type, dtype=torch.float16, enabled=(self.device==use_autocast)):
            # Generate speech tokens with T3
            speech_tokens = self.t3.inference(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=1000,
                temperature=temperature,
                cfg_weight=cfg_weight,
                repetition_penalty=repetition_penalty,
                top_p=top_p,
            )
            
            # Extract only the conditional batch and filter invalid tokens
            speech_tokens = speech_tokens[0]
            speech_tokens = drop_invalid_tokens(speech_tokens)
            
            # FIX (Root Cause): Remove the last TWO tokens, not just one. The
            # final 1-2 T3 speech tokens decode to a high-ZCR buzzing burst that
            # S3Gen renders as an audible "rè rè" tail right before each pause.
            if len(speech_tokens) > 3:
                speech_tokens = speech_tokens[:-2]
                
            speech_tokens = speech_tokens.to(self.device)
        
            # Generate waveform with S3Gen
            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.conds.s3,
            )
            
        return wav[0].cpu().numpy()
    
    def generate(
        self,
        text: str,
        language: str = "vi",
        audio_prompt: Optional[Union[str, Path, torch.Tensor]] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,

        top_p: float = 1.0,
        repetition_penalty: float = 2.0,
        split_sentences: bool = True,
        crossfade_ms: int = 50,
        sentence_pause_ms: int = 500,
        prompt_time_stretch: float = 1.0,
        cond_margin_s: float = 0.12,
    ) -> torch.Tensor:
        """
        Generate speech from text.
        
        Args:
            text: Input text to synthesize
            language: Language code ('vi' or 'en')
            audio_prompt: Optional reference audio for voice cloning
            exaggeration: Expression intensity (0.0 - 2.0)
            cfg_weight: Classifier-free guidance weight (0.0 - 1.0)
            temperature: Sampling temperature (0.1 - 1.0)
            top_p: Top-p sampling parameter
            repetition_penalty: Repetition penalty for T3
            split_sentences: Whether to split text by punctuation and generate separately
            crossfade_ms: Crossfade duration in milliseconds when merging sentences
            sentence_pause_ms: Pause duration between sentences in milliseconds (default 500ms)
            prompt_time_stretch: Tempo applied to the T3 speech-conditioning prompt
                (timbre-preserving pacing control; see ``prepare_conditionals``).
            cond_margin_s: VAD trim margin kept after the last speech frame;
                a generous margin (>= 0.12 s) prevents short titles like
                "Phần dẫn." from losing their falling tail.
            
        Returns:
            Audio tensor (1, samples) at 24kHz
        """
        # Prepare conditioning - use random voice if no audio_prompt and no conds
        if audio_prompt is not None:
            self.prepare_conditionals(audio_prompt, exaggeration, prompt_time_stretch)
        elif self.conds is None:
            # Try to use a random voice from wavs folder
            random_voice = get_random_voice()
            if random_voice is not None:
                self.prepare_conditionals(random_voice, exaggeration)
            else:
                raise ValueError("No reference audio! Add .wav files to wavs/ folder or provide audio_prompt.")
        
        # Normalize text (convert numbers, abbreviations to words for Vietnamese)
        text = normalize_text(text, language)
        
        if split_sentences:
            # Split text into sentences
            sentences = _split_text_to_sentences(text)
            
            if len(sentences) == 0:
                sentences = [text]
            elif len(sentences) == 1:
                # Single sentence, no need for splitting logic
                pass
            
            # Generate each sentence
            audio_segments = []
            for i, sentence in enumerate(sentences):
                print(f"  [{i+1}/{len(sentences)}] {sentence[:50]}...")
                
                audio_np = self._generate_single(
                    text=sentence,
                    language=language,
                    cfg_weight=cfg_weight,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )
                
                # Trim silence using VAD (more precise endpointing)
                # Keep a generous margin so short titles ("Phần dẫn.") do not
                # lose their falling tail to an aggressive VAD endpoint.
                audio_np = vad_trim(audio_np, self.sr, margin_s=cond_margin_s)

                # Remove the buzzing S3Gen tail artifact (high-ZCR noise island
                # after the silence valley) and fade to exactly 0.0.
                audio_np = tail_cleanup_clean(audio_np, self.sr)

                # Apply fade-out to prevent click at end of each segment
                audio_np = apply_fade_out(audio_np, self.sr, fade_duration=0.01)  # 10ms fade-out
                
                # Apply fade-in to prevent click at start
                audio_np = apply_fade_in(audio_np, self.sr, fade_duration=0.005)  # 5ms fade-in
                
                if len(audio_np) > 0:
                    audio_segments.append(audio_np)
            
            # Merge with crossfading and pause
            if audio_segments:
                merged = crossfade_concat(audio_segments, self.sr, fade_ms=crossfade_ms, pause_ms=sentence_pause_ms)
                
                # Apply final fade-out to prevent click at very end
                merged = apply_fade_out(merged, self.sr, fade_duration=0.015)  # 15ms fade-out
                
                return torch.from_numpy(merged).unsqueeze(0)
            else:
                return torch.zeros(1, self.sr)  # 1 second of silence as fallback
        else:
            # Single generation without splitting
            audio_np = self._generate_single(
                text=text,
                language=language,
                cfg_weight=cfg_weight,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            # Remove the buzzing tail artifact S3Gen leaves after the real
            # speech decays to silence, then fade to exactly 0.0 so the clause
            # joins the following pause click-free.
            audio_np = tail_cleanup_clean(audio_np, self.sr)
            return torch.from_numpy(audio_np).unsqueeze(0)
    
    def save_audio(self, audio: torch.Tensor, path: Union[str, Path], trim_silence: bool = True):
        """
        Save audio to file.
        
        Args:
            audio: Audio tensor from generate()
            path: Output file path
            trim_silence: Whether to trim trailing silence
        """
        import soundfile as sf
        
        audio_np = audio[0].cpu().numpy()
        
        if trim_silence:
            audio_np, _ = librosa.effects.trim(audio_np, top_db=30)
        
        sf.write(str(path), audio_np, self.sr)
