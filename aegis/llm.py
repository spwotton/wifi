from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    def load_dotenv(*args, **kwargs):  # type: ignore
        return False

import requests


@dataclass
class LLMConfig:
    provider: str = "openai"  # openai|gemini|hf
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens: int = 300


def _env(key: str) -> Optional[str]:
    # Try local env files first, then system env
    try:
        load_dotenv(dotenv_path=".env.local", override=False)
    except Exception:
        pass
    try:
        load_dotenv(override=False)
    except Exception:
        pass
    return os.environ.get(key)


def _post(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: int = 15) -> Dict[str, Any]:
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()
def _openai_base_url() -> str:
    # Allow overriding OpenAI endpoint to target local OpenAI-compatible servers (e.g., Ollama/llama.cpp gateways)
    return (
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("LLM_BASE_URL")
        or os.environ.get("OLLAMA_BASE_URL")
        or "https://api.openai.com/v1"
    ).rstrip("/")


def _requires_openai_key(base_url: str) -> bool:
    return "openai.com" in base_url.lower()



def _hf_client():
    try:
        from huggingface_hub import InferenceClient  # type: ignore
    except Exception:
        return None
    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_TOKEN")
    if not token:
        return None
    try:
        return InferenceClient(provider="hf-inference", api_key=token)
    except Exception:
        return None


# --- Optional llama.cpp backend -------------------------------------------------
_LLAMACPP_SINGLETON = None  # cache model within process


def _llamacpp_load_model(model_spec: str):
    """Load a llama.cpp model based on a spec.

    Supported formats for model_spec:
      - Local GGUF path, e.g., "C:/models/MyModel.gguf"
      - Repo form: "repo:owner/name#filename.gguf" (downloads via from_pretrained)

    If not provided or invalid, fall back to env:
      - LLMACPP_MODEL_PATH (local path)
      - LLMACPP_REPO_ID + LLMACPP_FILENAME (from_pretrained)
    """
    global _LLAMACPP_SINGLETON
    if _LLAMACPP_SINGLETON is not None:
        return _LLAMACPP_SINGLETON
    try:
        from llama_cpp import Llama  # type: ignore
    except Exception as e:
        raise RuntimeError("llama-cpp-python not installed. pip install llama-cpp-python") from e

    n_ctx = int(os.environ.get("LLMACPP_CTX", "4096"))
    n_threads = int(os.environ.get("LLMACPP_THREADS", "0"))  # 0 = auto
    # First try model_spec
    path_candidate = model_spec or ""
    # If looks like repo:...
    if path_candidate.startswith("repo:"):
        frag = path_candidate[len("repo:"):]
        repo_id, filename = (frag.split("#", 1) + [""])[:2]
        if not filename:
            filename = os.environ.get("LLMACPP_FILENAME", "")
        if not repo_id or not filename:
            raise RuntimeError("llamacpp repo spec requires 'repo:owner/name#filename.gguf'")
        llm = Llama.from_pretrained(repo_id=repo_id, filename=filename, n_ctx=n_ctx, n_threads=n_threads)
        _LLAMACPP_SINGLETON = llm
        return llm
    # Local path
    if path_candidate and os.path.exists(path_candidate):
        llm = Llama(model_path=path_candidate, n_ctx=n_ctx, n_threads=n_threads)
        _LLAMACPP_SINGLETON = llm
        return llm
    # Env fallbacks
    env_path = os.environ.get("LLMACPP_MODEL_PATH", "")
    if env_path and os.path.exists(env_path):
        llm = Llama(model_path=env_path, n_ctx=n_ctx, n_threads=n_threads)
        _LLAMACPP_SINGLETON = llm
        return llm
    repo_id = os.environ.get("LLMACPP_REPO_ID", "")
    filename = os.environ.get("LLMACPP_FILENAME", "")
    if repo_id and filename:
        llm = Llama.from_pretrained(repo_id=repo_id, filename=filename, n_ctx=n_ctx, n_threads=n_threads)
        _LLAMACPP_SINGLETON = llm
        return llm
    raise RuntimeError(
        "llama.cpp not configured. Provide a local GGUF path as model, use 'repo:owner/name#file.gguf', or set LLMACPP_MODEL_PATH or LLMACPP_REPO_ID+LLMACPP_FILENAME."
    )


def _llamacpp_chat(system_prompt: str, user_text: str, model_spec: str, temperature: float, max_tokens: int) -> str:
    llm = _llamacpp_load_model(model_spec)
    try:
        # Prefer chat API if available
        out = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # llama.cpp returns choices list similar to OpenAI
        return (
            out.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
    except Exception:
        # Fallback to plain generation
        combined = system_prompt + "\n\n" + user_text
        out = llm(combined, max_tokens=max_tokens, temperature=temperature)
        txt = out["choices"][0]["text"] if isinstance(out, dict) else str(out)
        return str(txt).strip()


def summarize_markdown(md_text: str, cfg: LLMConfig) -> str:
    provider = cfg.provider.lower()
    if provider == "openai":
        api_key = _env("OPENAI_API_KEY")
        base_url = _openai_base_url()
        if _requires_openai_key(base_url) and not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        url = f"{base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        prompt = "Summarize the following security incident report into a concise executive summary with 3 bullets and one recommended next action. Keep it under 120 words.\n\n" + md_text
        payload = {
            "model": cfg.model,
            "messages": [
                {"role": "system", "content": "You are a helpful security analyst."},
                {"role": "user", "content": prompt},
            ],
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
        }
        data = _post(url, headers, payload)
        return data["choices"][0]["message"]["content"].strip()
    elif provider == "gemini":
        api_key = _env("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set")
        model = cfg.model or "gemini-1.5-flash"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        prompt = "Summarize the following security incident report into a concise executive summary with 3 bullets and one recommended next action. Keep it under 120 words.\n\n" + md_text
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        data = _post(url, {"Content-Type": "application/json"}, payload)
        # Gemini returns candidates -> content -> parts[0].text
        cand = data.get("candidates", [{}])[0]
        content = cand.get("content", {})
        parts = content.get("parts", [{}])
        text = parts[0].get("text", "").strip()
        return text
    elif provider == "hf":
        model = cfg.model or "sshleifer/distilbart-cnn-12-6"
        client = _hf_client()
        prompt = ("Summarize: " + md_text[:2000])
        if client is not None:
            try:
                # Many HF summarization models use text-generation interface; client.text_generation covers generic
                data = client.text_generation(prompt, model=model, max_new_tokens=min(cfg.max_tokens, 200), temperature=cfg.temperature)
                if isinstance(data, dict) and "generated_text" in data:
                    return str(data["generated_text"]).strip()
                if isinstance(data, str):
                    return data.strip()
            except Exception:
                pass
        token = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN or HF_API_TOKEN not set")
        url = f"https://api-inference.huggingface.co/models/{model}"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": min(cfg.max_tokens, 200), "temperature": cfg.temperature}}
        data = _post(url, headers, payload)
        if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
            return str(data[0]["generated_text"]).strip()
        if isinstance(data, dict) and "error" in data:
            raise RuntimeError(f"HF error: {data['error']}")
        return json.dumps(data)
    elif provider == "llamacpp":
        # Use local llama.cpp model; cfg.model can be a GGUF path or repo:owner/name#file.gguf
        prompt = (
            "Summarize the following security incident report into a concise executive summary with 3 bullets and one recommended next action. Keep it under 120 words.\n\n"
            + md_text
        )
        return _llamacpp_chat("You are a helpful security analyst.", prompt, cfg.model, cfg.temperature, cfg.max_tokens)
    else:
        raise ValueError(f"Unknown provider: {cfg.provider}")


def chat_text(prompt: str, cfg: LLMConfig, system_prompt: str = "You are a helpful assistant.") -> str:
    provider = cfg.provider.lower()
    if provider == "openai":
        api_key = _env("OPENAI_API_KEY")
        base_url = _openai_base_url()
        if _requires_openai_key(base_url) and not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        url = f"{base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": cfg.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
        }
        data = _post(url, headers, payload)
        return data["choices"][0]["message"]["content"].strip()
    elif provider == "gemini":
        api_key = _env("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set")
        model = cfg.model or "gemini-1.5-flash"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        # Combine system + user as text parts; Gemini supports safety/system via roles in v1beta2, here emulate
        payload = {"contents": [{"parts": [{"text": system_prompt + "\n\n" + prompt}]}]}
        data = _post(url, {"Content-Type": "application/json"}, payload)
        cand = data.get("candidates", [{}])[0]
        content = cand.get("content", {})
        parts = content.get("parts", [{}])
        return parts[0].get("text", "").strip()
    elif provider == "hf":
        model = cfg.model or "facebook/bart-large-cnn"
        client = _hf_client()
        payload_text = system_prompt + "\n\n" + prompt
        if client is not None:
            try:
                data = client.text_generation(payload_text, model=model, max_new_tokens=min(cfg.max_tokens, 256), temperature=cfg.temperature)
                if isinstance(data, dict) and "generated_text" in data:
                    return str(data["generated_text"]).strip()
                if isinstance(data, str):
                    return data.strip()
            except Exception:
                pass
        token = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN or HF_API_TOKEN not set")
        url = f"https://api-inference.huggingface.co/models/{model}"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        payload = {"inputs": payload_text, "parameters": {"max_new_tokens": min(cfg.max_tokens, 256), "temperature": cfg.temperature}}
        data = _post(url, headers, payload)
        if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
            return str(data[0]["generated_text"]).strip()
        if isinstance(data, dict) and "error" in data:
            raise RuntimeError(f"HF error: {data['error']}")
        return json.dumps(data)
    elif provider == "llamacpp":
        return _llamacpp_chat(system_prompt, prompt, cfg.model, cfg.temperature, cfg.max_tokens)
    else:
        raise ValueError(f"Unknown provider: {cfg.provider}")


def chat_text_stream(prompt: str, cfg: LLMConfig, system_prompt: str = "You are a helpful assistant."):
    """Yield chunks of text for streaming-capable providers.

    Supports:
      - openai-compatible chat/completions with stream=true
      - llamacpp via create_chat_completion(stream=True)

    For non-streaming providers, yields one chunk with the full output.
    """
    provider = cfg.provider.lower()
    if provider == "openai":
        api_key = _env("OPENAI_API_KEY")
        base_url = _openai_base_url()
        if _requires_openai_key(base_url) and not api_key:
            # Fall back to non-stream error
            yield "[OPENAI] Missing OPENAI_API_KEY"
            return
        url = f"{base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": cfg.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "stream": True,
        }
        # Use streaming via requests; iterate over SSE lines
        with requests.post(url, headers=headers, json=payload, stream=True) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[len("data: ") :].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                        delta = obj.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if delta:
                            yield delta
                    except Exception:
                        continue
        return
    if provider == "llamacpp":
        llm = _llamacpp_load_model(cfg.model)
        try:
            it = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                stream=True,
            )
            for part in it:
                try:
                    delta = (
                        part.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content", "")
                    )
                    if delta:
                        yield delta
                except Exception:
                    continue
            return
        except Exception:
            # Fallback to non-stream
            yield chat_text(prompt, cfg, system_prompt)
            return
    # Default: non-stream full output as one chunk
    yield chat_text(prompt, cfg, system_prompt)


def quick_test() -> dict:
    """Attempt a very small request to any providers with keys present. Returns status per provider."""
    results: dict[str, str] = {}
    md = "Test: Detect and report a single-sentence summary: The system observed coincident ultrasonic activity and Wi‑Fi deauthentication spikes."
    for provider, model in (
        ("openai", "gpt-4o-mini"),
        ("gemini", "gemini-1.5-flash"),
        ("hf", "sshleifer/distilbart-cnn-12-6"),
    ):
        try:
            if provider == "openai" and not _env("OPENAI_API_KEY"):
                results[provider] = "SKIPPED (no OPENAI_API_KEY)"
                continue
            if provider == "gemini" and not _env("GOOGLE_API_KEY"):
                results[provider] = "SKIPPED (no GOOGLE_API_KEY)"
                continue
            if provider == "hf" and not _env("HF_API_TOKEN"):
                results[provider] = "SKIPPED (no HF_API_TOKEN)"
                continue
            out = summarize_markdown(md, LLMConfig(provider=provider, model=model, max_tokens=64))
            results[provider] = "OK"
        except Exception as e:
            results[provider] = f"ERROR: {e}"[:200]
    return results


def prefer_and_summarize(md_text: str) -> tuple[str, str, str]:
    """Try providers/models in a preferred order and return (summary, provider, model).

    Order reflects user's preference:
      1) Gemini Pro latest (attempt 2.5/2.0/1.5),
      2) OpenAI (attempt GPT-5/4.1/4o/4o-mini),
      3) HF summarization model as last resort.
    """
    preferences: list[tuple[str, list[str]]] = [
        ("gemini", [
            "gemini-2.5-pro",  # attempt if available
            "gemini-2.0-pro",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        ]),
        ("openai", [
            "gpt-5",         # attempt if available
            "gpt-4.1",
            "gpt-4o",
            "gpt-4o-mini",
        ]),
        ("hf", [
            "sshleifer/distilbart-cnn-12-6",
        ]),
        ("llamacpp", [
            os.environ.get("LLMACPP_MODEL_PATH", "") or os.environ.get("LLMACPP_REPO_ID", "repo:") + "#" + os.environ.get("LLMACPP_FILENAME", "")
        ]),
    ]
    last_err: Optional[Exception] = None
    for provider, models in preferences:
        for model in models:
            try:
                out = summarize_markdown(md_text, LLMConfig(provider=provider, model=model, max_tokens=256))
                return out, provider, model
            except Exception as e:
                last_err = e
                continue
    raise RuntimeError(f"All provider attempts failed: {last_err}")


def prefer_and_chat(user_prompt: str, system_prompt: str = "You are a helpful assistant.") -> Tuple[str, str, str]:
    """Auto preference for chat, mirroring summarize order.

    Returns (answer, provider, model).
    """
    preferences: list[tuple[str, list[str]]] = [
        ("gemini", [
            "gemini-2.5-pro",
            "gemini-2.0-pro",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        ]),
        ("openai", [
            "gpt-5",
            "gpt-4.1",
            "gpt-4o",
            "gpt-4o-mini",
        ]),
        ("hf", [
            "facebook/bart-large-cnn",
        ]),
        ("llamacpp", [
            os.environ.get("LLMACPP_MODEL_PATH", "") or os.environ.get("LLMACPP_REPO_ID", "repo:") + "#" + os.environ.get("LLMACPP_FILENAME", "")
        ]),
    ]
    last_err: Optional[Exception] = None
    for provider, models in preferences:
        for model in models:
            try:
                out = chat_text(user_prompt, LLMConfig(provider=provider, model=model, max_tokens=512), system_prompt=system_prompt)
                return out, provider, model
            except Exception as e:
                last_err = e
                continue
    raise RuntimeError(f"All provider attempts failed: {last_err}")
