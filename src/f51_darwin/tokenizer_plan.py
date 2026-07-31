from __future__ import annotations


class SimpleByteTokenizer:
    """Byte-level tokenizer for toy runtime only.

    The production plan is a F51-owned SentencePiece or BPE tokenizer trained
    from the project corpus. This toy tokenizer keeps the seed training script
    independent of external model assets.
    """

    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids: list[int]) -> str:
        return bytes(int(token) % 256 for token in token_ids).decode("utf-8", errors="replace")


TOKENIZER_PLAN = {
    "v0_toy": "SimpleByteTokenizer for import, tests and toy overfit proof.",
    "v1_seed": "Train F51-owned SentencePiece or byte-pair tokenizer from curated F51 corpus.",
    "forbidden": "No tokenizer copied from Llama, Qwen, Mistral, GPT, Phi, Gemma or DeepSeek.",
}

