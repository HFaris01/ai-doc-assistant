from __future__ import annotations

import os

from ollama import Client


OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:latest")

INSUFFICIENT_CONTEXT_MESSAGE = (
    "I could not find a reliable answer in the provided document context."
)


def get_ollama_client() -> Client:
    return Client(host=OLLAMA_HOST)


def _format_context_chunks(context_chunks: list[str]) -> str:
    cleaned_chunks = []

    for i, chunk in enumerate(context_chunks, start=1):
        chunk = chunk.strip()

        if not chunk:
            continue

        cleaned_chunks.append(f"[Context {i}]\n{chunk}")

    return "\n\n".join(cleaned_chunks)


def build_grounded_messages(query: str, context_chunks: list[str]) -> list[dict]:
    context_text = _format_context_chunks(context_chunks)

    system_prompt = (
        "You are a document question-answering assistant.\n"
        "Answer only from the provided document context.\n"
        "Do not use outside knowledge.\n"
        f"If the context is insufficient, say: '{INSUFFICIENT_CONTEXT_MESSAGE}'\n"
        "Keep the answer concise, clear, and directly related to the question.\n"
        "Do not invent facts.\n"
    )

    user_prompt = (
        f"Question:\n{query.strip()}\n\n"
        f"Document context:\n{context_text}\n\n"
        "Write a grounded answer based only on the context above."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _extract_response_text(response) -> str:
    if hasattr(response, "message") and hasattr(response.message, "content"):
        return response.message.content.strip()

    if isinstance(response, dict):
        message = response.get("message", {})

        if isinstance(message, dict):
            return message.get("content", "").strip()

    return ""


def generate_grounded_answer_ollama(
    query: str,
    context_chunks: list[str],
    model_name: str = DEFAULT_OLLAMA_MODEL,
) -> str:
    if not query.strip():
        return "Please enter a question."

    if not any(chunk.strip() for chunk in context_chunks):
        return INSUFFICIENT_CONTEXT_MESSAGE

    client = get_ollama_client()
    messages = build_grounded_messages(query, context_chunks)

    response = client.chat(
        model=model_name,
        messages=messages,
    )

    answer = _extract_response_text(response)

    if not answer:
        return "I could not generate an answer from the local model."

    return answer


def _extract_model_names(models_response) -> list[str]:
    model_names = []

    if hasattr(models_response, "models"):
        for model in models_response.models:
            if hasattr(model, "model"):
                model_names.append(model.model)

    elif isinstance(models_response, dict) and "models" in models_response:
        for model in models_response["models"]:
            if isinstance(model, dict) and "model" in model:
                model_names.append(model["model"])

    return model_names


def _is_model_available(model_name: str, model_names: list[str]) -> bool:
    if model_name in model_names:
        return True

    # Ollama often lists models with tags, for example gemma3:latest.
    # This lets a user type "gemma3" while the local model is stored as "gemma3:latest".
    if ":" not in model_name:
        tagged_name = f"{model_name}:latest"
        return tagged_name in model_names

    return False


def check_ollama_connection(model_name: str = DEFAULT_OLLAMA_MODEL) -> tuple[bool, str]:
    try:
        client = get_ollama_client()
        models_response = client.list()
        model_names = _extract_model_names(models_response)

        if not model_names:
            return False, "Ollama is reachable, but no local models were found."

        if not _is_model_available(model_name, model_names):
            available_models = ", ".join(model_names)
            return (
                False,
                f"Ollama is reachable, but model '{model_name}' was not found. "
                f"Available models: {available_models}",
            )

        return True, "Ollama is running and the model is available."

    except Exception as e:
        return False, f"Could not connect to Ollama: {e}"