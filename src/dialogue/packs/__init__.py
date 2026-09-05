"""Prompt packs for ``build_chatbot_system_prompt``.

A "pack" is a plain module of string constants supplying the vertical-specific
SCOPE text and illustrative example phrases that ``build_chatbot_system_prompt``
(in ``src.dialogue.prompts``) assembles into the ChatBot system prompt. Packs
hold no logic — selection, fallback-on-unknown-key, and text assembly all live
in ``prompts.py``.

Add a new vertical by adding a new module here (mirroring the constant names
in ``betting.py``/``generic.py``) and registering it in ``prompts.PACKS``.
"""
