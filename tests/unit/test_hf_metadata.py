import json
from types import SimpleNamespace

from armcc.hf_metadata import HFMetadataExtractor, _token_id_list


def test_normalizes_scalar_and_multiple_special_token_ids():
    assert _token_id_list(2, 0) == [2]
    assert _token_id_list([2, 3, 2], 0) == [2, 3, 2]
    assert _token_id_list([], 2) == [2]
    assert _token_id_list(None, 2) == [2]


def test_metadata_keeps_all_eos_tokens_and_selects_a_primary_token(tmp_path):
    class Tokenizer:
        chat_template = "{{ messages }}"

        def save_pretrained(self, path):
            (tmp_path / "tokenizer").mkdir(exist_ok=True)

    loaded = SimpleNamespace(
        config=SimpleNamespace(bos_token_id=1, eos_token_id=[2, 3], pad_token_id=2),
        tokenizer=Tokenizer(),
        model_id="org/model",
        arch_family="llama_style",
        num_layers=1,
        hidden_size=16,
        num_heads=2,
        num_kv_heads=1,
        intermediate_size=32,
        vocab_size=128,
        max_position_embeddings=512,
        total_params=1,
        total_bytes=2,
    )

    runtime = HFMetadataExtractor(tmp_path).extract(loaded)

    assert runtime["eos_token_id"] == 2
    assert runtime["eos_token_ids"] == [2, 3]
    assert json.loads((tmp_path / "runtime_config.json").read_text())["eos_token_ids"] == [2, 3]
