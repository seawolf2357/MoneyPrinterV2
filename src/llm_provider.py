import os

from config import is_running_in_spaces

_selected_model: str | None = None


def _use_hf_backend() -> bool:
    """Use HF Inference API when running on Spaces or when HF_TOKEN is set and Ollama is absent."""
    if is_running_in_spaces():
        return True
    if os.environ.get("HF_TOKEN") and not os.environ.get("OLLAMA_BASE_URL"):
        return True
    return False


# ---------------------------------------------------------------------------
# HF Inference API backend
# ---------------------------------------------------------------------------

def _hf_client():
    from huggingface_hub import InferenceClient
    token = os.environ.get("HF_TOKEN", "")
    return InferenceClient(token=token)


def _hf_list_models() -> list[str]:
    return [
        "meta-llama/Llama-3.1-8B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "google/gemma-2-9b-it",
    ]


def _hf_generate_text(prompt: str, model: str) -> str:
    response = _hf_client().chat_completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Ollama backend (original)
# ---------------------------------------------------------------------------

def _ollama_client():
    import ollama
    from config import get_ollama_base_url
    return ollama.Client(host=get_ollama_base_url())


def _ollama_list_models() -> list[str]:
    response = _ollama_client().list()
    return sorted(m.model for m in response.models)


def _ollama_generate_text(prompt: str, model: str) -> str:
    response = _ollama_client().chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Public API (unchanged interface)
# ---------------------------------------------------------------------------

def list_models() -> list[str]:
    if _use_hf_backend():
        return _hf_list_models()
    return _ollama_list_models()


def select_model(model: str) -> None:
    global _selected_model
    _selected_model = model


def get_active_model() -> str | None:
    return _selected_model


def generate_text(prompt: str, model_name: str = None) -> str:
    model = model_name or _selected_model
    if not model:
        raise RuntimeError(
            "No model selected. Call select_model() first or pass model_name."
        )

    if _use_hf_backend():
        return _hf_generate_text(prompt, model)
    return _ollama_generate_text(prompt, model)
