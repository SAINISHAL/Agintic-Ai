from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from src.chatbot.context import build_context
from src.chatbot.prompt_builder import build_messages
from src.chatbot.safety import grounding_used, safety_check
from src.emotion.baseline import keyword_predict
from src.emotion.inference import EmotionClassifier
from src.qwen.inference import QwenGenerator
from src.utils.config import load_config, project_root, resolve_path
from src.utils.device import select_device
from src.utils.emotions import CANONICAL_EMOTIONS

logger = logging.getLogger(__name__)

# Supported generation providers
_PROVIDER_QWEN = "qwen"
_PROVIDER_MENTAL_LLM = "mental_llm"

_UNPREPARED_PATTERN = re.compile(
    r"\b(?:0|zero|no)\s+(?:preparation|prep)\b|\b(?:not|un)prepared\b|\b(?:haven't|have not)\s+prepared\b",
    re.I,
)
_FALSE_PREPARED_CLAIM = re.compile(
    r"\b(?:you(?:'re| are)\s+(?:already\s+|fully\s+)?prepared|already\s+prepared)\b",
    re.I,
)


def correct_preparation_contradiction(user_message: str, response: str) -> str:
    if _UNPREPARED_PATTERN.search(user_message) and _FALSE_PREPARED_CLAIM.search(response):
        return (
            "That sounds stressful, especially with placements coming up and feeling unprepared. "
            "Start with one manageable step: choose a role and practice one common interview question. "
            "I can help you make a short preparation plan."
        )
    return response


class CounsellingChatbot:
    """Emotion-aware counselling chatbot with optional RAG grounding.

    Supports multiple generation providers:
        - ``qwen``       — existing Qwen + LoRA pipeline
        - ``mental_llm`` — mental-health-focused pretrained LLM

    Pipeline:
        user message
        → safety check (crisis takes priority — RAG never overrides this)
        → emotion classification
        → context / need detection
        → RAG retrieval (if enabled and not a crisis)
        → prompt construction (with retrieved evidence)
        → selected generator (Qwen *or* MentalHealth LLM)
        → post-processing
    """

    def __init__(
        self,
        emotion_model_path: str | Path | None = None,
        qwen_model_path: str | Path | None = None,
        chatbot_config_path: str | Path | None = None,
    ) -> None:
        self.root = project_root()
        cfg_path = chatbot_config_path or (self.root / "configs" / "chatbot.yaml")
        self.config = load_config(cfg_path)
        self.emotion_model_path = Path(emotion_model_path) if emotion_model_path else resolve_path(
            self.config["emotion"]["model_path"], self.root
        )
        self.qwen_model_path = Path(qwen_model_path) if qwen_model_path else resolve_path(
            self.config["qwen"]["model_path"], self.root
        )
        self.session_history: list[dict[str, str]] = []
        self.emotion_model: EmotionClassifier | None = None
        self.generator: QwenGenerator | Any | None = None  # QwenGenerator or MentalHealthGenerator
        self._provider: str = self.config.get("generation", {}).get("provider", _PROVIDER_QWEN)
        self.max_turns = int(self.config.get("history", {}).get("max_turns", 12))
        self._initialized = False

        # RAG pipeline — lazy-initialised in initialize()
        self._rag: Any | None = None  # RAGPipeline | None
        self._rag_enabled = False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        select_device()

        # --- Emotion model ---
        if self.emotion_model_path.exists() and (self.emotion_model_path / "label_mapping.json").exists():
            self.emotion_model = EmotionClassifier(self.emotion_model_path)
        elif self.config["emotion"].get("fallback_to_baseline", True):
            print(
                f"Emotion checkpoint not found at {self.emotion_model_path}. "
                "Using keyword baseline until you run scripts/train_emotion.py."
            )
        else:
            raise FileNotFoundError(f"Emotion model not found: {self.emotion_model_path}")

        # --- Generator (dispatch by provider) ---
        if self._provider == _PROVIDER_MENTAL_LLM:
            self._init_mental_llm()
        else:
            self._init_qwen()

        # --- RAG pipeline ---
        self._init_rag()

        self._initialized = True

    def _init_qwen(self) -> None:
        """Initialise the Qwen generator (existing logic, unchanged)."""
        qwen_cfg = load_config(self.root / self.config["qwen"]["config_path"])
        gen_cfg = qwen_cfg.get("generation", {})

        adapter_or_model = self.qwen_model_path
        base_name = self.config["qwen"].get("base_model_name", "Qwen/Qwen3-0.6B")
        load_4bit = bool(self.config["qwen"].get("load_in_4bit", False))
        use_finetuned = bool(self.config["qwen"].get("use_finetuned", False))
        inference_backend = self.config.get("qwen", {}).get("inference_backend", "local")
        fallback_base = self.config["qwen"].get("fallback_base_name", "Qwen/Qwen3-0.6B")

        if not use_finetuned:
            print(f"Local chatbot mode: prompting-only {base_name} (backend={inference_backend}).")
            self.generator = QwenGenerator(
                model_path=base_name,
                base_model_name=base_name,
                load_in_4bit=load_4bit,
                generation_config=gen_cfg,
                inference_backend=inference_backend,
            )
        elif adapter_or_model.exists() and (
            (adapter_or_model / "adapter_config.json").exists() or (adapter_or_model / "config.json").exists()
        ):
            try:
                print(f"Attempting to load fine-tuned model from {adapter_or_model} with base {base_name}...")
                self.generator = QwenGenerator(
                    model_path=adapter_or_model,
                    base_model_name=base_name,
                    load_in_4bit=load_4bit,
                    generation_config=gen_cfg,
                    inference_backend=inference_backend,
                )
            except Exception as err:
                if self.config["qwen"].get("fallback_to_base", True):
                    print(
                        f"[NOTICE] Could not load fine-tuned adapter locally ({err}).\n"
                        f"Falling back to local CPU base model: {fallback_base}"
                    )
                    self.generator = QwenGenerator(
                        model_path=fallback_base,
                        base_model_name=fallback_base,
                        load_in_4bit=False,
                        generation_config=gen_cfg,
                        inference_backend=inference_backend,
                    )
                else:
                    raise
        elif self.config["qwen"].get("fallback_to_base", True):
            print(
                f"Fine-tuned Qwen path not found at {adapter_or_model}. "
                f"Loading base model {fallback_base} as the local baseline."
            )
            self.generator = QwenGenerator(
                model_path=fallback_base,
                base_model_name=fallback_base,
                load_in_4bit=False,
                generation_config=gen_cfg,
                inference_backend=inference_backend,
            )
        else:
            raise FileNotFoundError(f"Qwen model not found: {adapter_or_model}")

    def _init_mental_llm(self) -> None:
        """Initialise the mental-health LLM generator."""
        ml_cfg = self.config.get("mental_llm", {})
        model_id = ml_cfg.get("model_id")
        if not model_id:
            raise ValueError(
                "generation.provider is 'mental_llm' but mental_llm.model_id is not set. "
                "Check configs/chatbot.yaml."
            )

        gen_cfg = {
            "max_new_tokens": ml_cfg.get("max_new_tokens", 256),
            "temperature": ml_cfg.get("temperature", 0.7),
            "top_p": ml_cfg.get("top_p", 0.9),
            "repetition_penalty": ml_cfg.get("repetition_penalty", 1.05),
        }

        try:
            from src.mental_llm.inference import MentalHealthGenerator  # noqa: PLC0415

            print(f"[Chatbot] Using mental_llm provider: {model_id}")
            self.generator = MentalHealthGenerator(
                model_id=model_id,
                device=ml_cfg.get("device", "auto"),
                torch_dtype=ml_cfg.get("torch_dtype", "auto"),
                generation_config=gen_cfg,
                trust_remote_code=ml_cfg.get("trust_remote_code", False),
            )
        except Exception as exc:
            logger.error(
                "[Chatbot] Failed to load mental_llm '%s': %s", model_id, exc,
                exc_info=True,
            )
            # Fall back to Qwen if configured
            if self.config.get("qwen", {}).get("fallback_to_base", True):
                print(
                    f"[NOTICE] mental_llm provider failed ({exc}).\n"
                    "Falling back to Qwen generator."
                )
                self._provider = _PROVIDER_QWEN
                self._init_qwen()
            else:
                raise

    def _init_rag(self) -> None:
        """Initialise the RAG pipeline from the chatbot config's ``rag`` section."""
        raw_rag_cfg = self.config.get("rag", {})
        if not raw_rag_cfg.get("enabled", True):
            logger.info("[Chatbot] RAG disabled in configuration.")
            self._rag_enabled = False
            return

        try:
            from src.rag.rag_pipeline import RAGPipeline  # noqa: PLC0415

            self._rag = RAGPipeline.from_config(raw_rag_cfg, project_root=self.root)
            self._rag_enabled = self._rag.is_ready
            if self._rag_enabled:
                logger.info("[Chatbot] RAG pipeline ready. Stats: %s", self._rag.stats)
            else:
                logger.warning(
                    "[Chatbot] RAG pipeline not ready: %s. Falling back to LLM-only.",
                    self._rag.init_error,
                )
        except ImportError:
            logger.warning(
                "[Chatbot] RAG dependencies not installed (sentence-transformers, chromadb). "
                "Falling back to LLM-only generation."
            )
            self._rag_enabled = False
        except Exception as exc:
            logger.error("[Chatbot] RAG initialisation failed: %s", exc, exc_info=True)
            self._rag_enabled = False

    # ------------------------------------------------------------------
    # Core pipeline steps
    # ------------------------------------------------------------------

    def detect_emotion(self, user_message: str) -> dict[str, Any]:
        text = f"[CURRENT]\nP: {user_message}"
        if self.emotion_model is not None:
            return self.emotion_model.predict(text)
        labels = keyword_predict(user_message, CANONICAL_EMOTIONS)
        return {
            "emotions": [{"label": lab, "confidence": 0.5} for lab in labels],
            "threshold": None,
            "probabilities": {},
            "backend": "keyword_baseline",
        }

    def build_context(
        self,
        user_message: str,
        emotions: list[dict[str, Any]],
        conversation_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        history = conversation_history if conversation_history is not None else self.session_history
        return build_context(user_message, history, emotions, max_turns=self.max_turns)

    def retrieve(
        self,
        user_message: str,
        emotions: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> list[Any]:
        """Run RAG retrieval. Returns empty list when RAG is unavailable."""
        if not self._rag_enabled or self._rag is None:
            return []
        emotion_labels = [e["label"] for e in emotions] if emotions else None
        need = context.get("need")
        try:
            return self._rag.retrieve(
                user_message,
                emotions=emotion_labels,
                need=need,
            )
        except Exception as exc:
            logger.warning("[Chatbot] RAG retrieval error: %s", exc)
            return []

    def generate_response(
        self,
        user_message: str,
        emotions: list[dict[str, Any]],
        context: dict[str, Any],
        conversation_history: list[dict[str, str]] | None = None,
        safety_note: str | None = None,
        retrieved: list[Any] | None = None,
    ) -> str:
        if self.generator is None:
            raise RuntimeError("Call initialize() before generate_response().")
        history = conversation_history if conversation_history is not None else self.session_history
        messages = build_messages(
            user_message, history, emotions, context,
            extra_safety_note=safety_note,
            retrieved=retrieved,
        )
        return self.generator.generate(messages)

    # ------------------------------------------------------------------
    # Main chat entry point
    # ------------------------------------------------------------------

    def chat(
        self,
        user_message: str,
        conversation_history: list[dict[str, str]] | None = None,
        persist: bool = True,
    ) -> dict[str, Any]:
        if not self._initialized:
            self.initialize()
        if conversation_history is None:
            conversation_history = list(self.session_history)

        # 1. Safety check — must be first; crisis takes absolute priority
        safety = safety_check(user_message)

        # 2. Emotion classification
        emotion_out = self.detect_emotion(user_message)
        emotions = emotion_out.get("emotions", [])

        # 3. Context / need detection
        context = self.build_context(user_message, emotions, conversation_history)

        # 4. RAG retrieval (skipped for crisis messages)
        retrieved: list[Any] = []
        if not safety.get("override_response"):
            retrieved = self.retrieve(user_message, emotions, context)

        # 5. Generate response
        if safety.get("override_response"):
            response = safety["override_response"]
        else:
            response = self.generate_response(
                user_message,
                emotions,
                context,
                conversation_history,
                safety_note=safety.get("prompt_note"),
                retrieved=retrieved,
            )
            response = correct_preparation_contradiction(user_message, response)

        # 6. Persist history
        if persist:
            self.session_history.append({"role": "user", "content": user_message})
            self.session_history.append({"role": "assistant", "content": response})
            if len(self.session_history) > self.max_turns * 2:
                self.session_history = self.session_history[-(self.max_turns * 2):]

        return {
            "response": response,
            "emotions": emotions,
            "context": context.get("need"),
            "grounding_used": grounding_used(response),
            "safety": {"crisis": safety["crisis"], "harm_to_others": safety["harm_to_others"]},
            "rag_sources": [
                {
                    "source": r.source_text,
                    "reference": r.chapter_reference,
                    "score": round(r.score, 3),
                }
                for r in retrieved
            ],
            "rag_enabled": self._rag_enabled,
            "provider": self._provider,
        }

    # ------------------------------------------------------------------
    # Convenience & compatibility methods
    # ------------------------------------------------------------------

    def respond(self, user_message: str) -> str:
        """Convenience method returning the text response directly."""
        res = self.chat(user_message, persist=True)
        return str(res.get("response", ""))

    def reset(self) -> None:
        """Reset conversation session history."""
        self.clear_history()

    def clear_history(self) -> None:
        self.session_history = []

    def generate_response_stream(
        self,
        user_message: str,
        emotions: list[dict[str, Any]],
        context: dict[str, Any],
        conversation_history: list[dict[str, str]] | None = None,
        safety_note: str | None = None,
        retrieved: list[Any] | None = None,
    ):
        """Stream response tokens as they become available."""
        if self.generator is None:
            raise RuntimeError("Call initialize() before generate_response_stream().")
        history = conversation_history if conversation_history is not None else self.session_history
        messages = build_messages(
            user_message, history, emotions, context,
            extra_safety_note=safety_note,
            retrieved=retrieved,
        )
        if hasattr(self.generator, "generate_stream"):
            yield from self.generator.generate_stream(messages)
        else:
            yield self.generator.generate(messages)

    def chat_stream(
        self,
        user_message: str,
        conversation_history: list[dict[str, str]] | None = None,
        persist: bool = True,
    ):
        """Stream response tokens progressively."""
        if not self._initialized:
            self.initialize()
        if conversation_history is None:
            conversation_history = list(self.session_history)

        safety = safety_check(user_message)
        emotion_out = self.detect_emotion(user_message)
        emotions = emotion_out.get("emotions", [])
        context = self.build_context(user_message, emotions, conversation_history)

        retrieved: list[Any] = []
        if not safety.get("override_response"):
            retrieved = self.retrieve(user_message, emotions, context)

        if safety.get("override_response"):
            override = safety["override_response"]
            yield override
            full_response = override
        else:
            collected = []
            for chunk in self.generate_response_stream(
                user_message,
                emotions,
                context,
                conversation_history,
                safety_note=safety.get("prompt_note"),
                retrieved=retrieved,
            ):
                collected.append(chunk)
                yield chunk
            full_response = "".join(collected)
            full_response = correct_preparation_contradiction(user_message, full_response)

        if persist:
            self.session_history.append({"role": "user", "content": user_message})
            self.session_history.append({"role": "assistant", "content": full_response})
            if len(self.session_history) > self.max_turns * 2:
                self.session_history = self.session_history[-(self.max_turns * 2):]
