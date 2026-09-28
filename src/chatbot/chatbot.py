from __future__ import annotations

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


class CounsellingChatbot:
    """Phase 1 chatbot: emotion → context → prompt → Qwen.

    Session history is in-memory only. No database, profiles, or embeddings.
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
        self.generator: QwenGenerator | None = None
        self.max_turns = int(self.config.get("history", {}).get("max_turns", 12))
        self._initialized = False

    def initialize(self) -> None:
        select_device()
        qwen_cfg = load_config(self.root / self.config["qwen"]["config_path"])
        gen_cfg = qwen_cfg.get("generation", {})

        if self.emotion_model_path.exists() and (self.emotion_model_path / "label_mapping.json").exists():
            self.emotion_model = EmotionClassifier(self.emotion_model_path)
        elif self.config["emotion"].get("fallback_to_baseline", True):
            print(
                f"Emotion checkpoint not found at {self.emotion_model_path}. "
                "Using keyword baseline until you run scripts/train_emotion.py."
            )
        else:
            raise FileNotFoundError(f"Emotion model not found: {self.emotion_model_path}")

        adapter_or_model = self.qwen_model_path
        base_name = self.config["qwen"].get("base_model_name", "Qwen/Qwen3-4B")
        load_4bit = bool(self.config["qwen"].get("load_in_4bit", True))
        if adapter_or_model.exists() and (
            (adapter_or_model / "adapter_config.json").exists() or (adapter_or_model / "config.json").exists()
        ):
            self.generator = QwenGenerator(
                model_path=adapter_or_model,
                base_model_name=base_name,
                load_in_4bit=load_4bit,
                generation_config=gen_cfg,
            )
        elif self.config["qwen"].get("fallback_to_base", True):
            print(
                f"Fine-tuned Qwen path not found at {adapter_or_model}. "
                f"Loading base model {base_name} as the Phase 1 baseline."
            )
            self.generator = QwenGenerator(
                model_path=base_name,
                base_model_name=base_name,
                load_in_4bit=load_4bit,
                generation_config=gen_cfg,
            )
        else:
            raise FileNotFoundError(f"Qwen model not found: {adapter_or_model}")
        self._initialized = True

    def detect_emotion(self, user_message: str, conversation_history: list[dict[str, str]] | None = None) -> dict[str, Any]:
        history = conversation_history if conversation_history is not None else self.session_history
        context_lines = []
        if history:
            context_lines.append("[CONTEXT]")
            for turn in history[-4:]:
                speaker = "P" if turn.get("role") == "user" else "T"
                context_lines.append(f"{speaker}: {turn.get('content', '')}")
        context_lines.append("[CURRENT]")
        context_lines.append(f"P: {user_message}")
        text = "\n".join(context_lines)
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

    def generate_response(
        self,
        user_message: str,
        emotions: list[dict[str, Any]],
        context: dict[str, Any],
        conversation_history: list[dict[str, str]] | None = None,
        safety_note: str | None = None,
    ) -> str:
        if self.generator is None:
            raise RuntimeError("Call initialize() before generate_response().")
        history = conversation_history if conversation_history is not None else self.session_history
        messages = build_messages(user_message, history, emotions, context, extra_safety_note=safety_note)
        return self.generator.generate(messages)

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

        safety = safety_check(user_message)
        emotion_out = self.detect_emotion(user_message, conversation_history)
        emotions = emotion_out.get("emotions", [])
        context = self.build_context(user_message, emotions, conversation_history)

        if safety.get("override_response"):
            response = safety["override_response"]
        else:
            response = self.generate_response(
                user_message,
                emotions,
                context,
                conversation_history,
                safety_note=safety.get("prompt_note"),
            )

        if persist:
            self.session_history.append({"role": "user", "content": user_message})
            self.session_history.append({"role": "assistant", "content": response})
            if len(self.session_history) > self.max_turns * 2:
                self.session_history = self.session_history[-(self.max_turns * 2) :]

        return {
            "response": response,
            "emotions": emotions,
            "context": context.get("need"),
            "grounding_used": grounding_used(response),
            "safety": {"crisis": safety["crisis"], "harm_to_others": safety["harm_to_others"]},
        }

    def clear_history(self) -> None:
        self.session_history = []
