"""ASR exports loaded on demand so text normalization needs no GPU dependencies."""
from importlib import import_module

_EXPORTS = {
    "config": ["ASRConfig", "get_asr_config"],
    "transcribe": ["SpeechTranscriber", "UnifiedASRTranscriber", "PhoWhisperEngine", "PhoASREngine", "FasterWhisperEngine"],
    "post_processor": ["LexiconPostProcessor", "ContextualPostProcessor", "PunctuationPostProcessor", "CompositePostProcessor"],
    "evaluate": ["ASREvaluator", "compute_wer", "compute_cer", "compute_token_f1"],
    "batch_runner": ["ASRBatchRunner"],
}
__all__ = [name for names in _EXPORTS.values() for name in names]


def __getattr__(name):
    for module, names in _EXPORTS.items():
        if name in names:
            value = getattr(import_module(f"src.asr.{module}"), name)
            globals()[name] = value
            return value
    raise AttributeError(name)