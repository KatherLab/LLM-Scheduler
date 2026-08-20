import textwrap

from app.catalog import load_catalog, merge_args


def _write(tmp_path, body: str) -> str:
    p = tmp_path / "models.yaml"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return str(p)


def test_defaults_applied_to_every_model(tmp_path):
    path = _write(
        tmp_path,
        """
        defaults:
          extra_args: "--enable-prompt-tokens-details"
          venv_activate: /shared/.venv/bin/activate
          cpus: 8
          env:
            HF_HUB_OFFLINE: "1"

        models:
          - name: a
            model_path: org/a
            gpus: 1
            tensor_parallel_size: 1
          - name: b
            model_path: org/b
            gpus: 2
            tensor_parallel_size: 2
            extra_args: "--enforce-eager"
            cpus: 32
            env:
              VLLM_FLAG: "2"
        """,
    )
    cat = load_catalog(path)

    a = cat["a"]
    assert a.extra_args == "--enable-prompt-tokens-details"
    assert a.venv_activate == "/shared/.venv/bin/activate"
    assert a.cpus == 8
    assert a.env == {"HF_HUB_OFFLINE": "1"}

    b = cat["b"]
    assert b.extra_args == "--enable-prompt-tokens-details --enforce-eager"
    assert b.cpus == 32
    assert b.env == {"HF_HUB_OFFLINE": "1", "VLLM_FLAG": "2"}


def test_catalog_without_defaults_is_unchanged(tmp_path):
    path = _write(
        tmp_path,
        """
        models:
          - name: a
            model_path: org/a
            gpus: 1
            tensor_parallel_size: 1
            extra_args: "--enforce-eager"
        """,
    )
    a = load_catalog(path)["a"]
    assert a.extra_args == "--enforce-eager"
    assert a.env is None
    assert a.cpus is None


def test_merge_args_model_overrides_same_flag():
    assert (
        merge_args(
            "--max-model-len 8192 --enable-prefix-caching", "--max-model-len 2048"
        )
        == "--enable-prefix-caching --max-model-len 2048"
    )
    assert (
        merge_args("--max-model-len 8192", "--max-model-len=2048")
        == "--max-model-len=2048"
    )
    assert merge_args("", "--enforce-eager") == "--enforce-eager"
    assert merge_args("--enforce-eager", "") == "--enforce-eager"


def test_merge_args_preserves_quoted_values():
    merged = merge_args('--chat-template "a b"', "--enforce-eager")
    assert merged == "--chat-template 'a b' --enforce-eager"
