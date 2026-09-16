"""Tests for ``scripts/fix_sarvam_tts_deprecated_config.py``.

Exercises ``compute_plan`` directly — it takes no DB dependency, so these run
with zero external dependencies. Mirrors the shape of
``tests/unit/test_crm_catalog_seeding.py`` for the sibling reseed script.
"""
from __future__ import annotations

from scripts.fix_sarvam_tts_deprecated_config import compute_plan
from src.providers.tts.sarvam import DEFAULT_MODEL, DEFAULT_SPEAKER


def _tenant(slug: str, tts: dict | None) -> dict:
    pc: dict = {"tts": tts} if tts is not None else {}
    return {"id": f"t_{slug}", "slug": slug, "pipeline_config": pc}


def test_pinned_v2_model_and_anushka_voice_are_fixed() -> None:
    tenants = [_tenant("example", {"provider": "sarvam", "model": "bulbul:v2",
                                    "voice_id": "anushka", "language": "hi-IN"})]
    plan = compute_plan(tenants)
    assert len(plan.fixes) == 1
    fix = plan.fixes[0]
    assert fix.slug == "example"
    assert fix.old_model == "bulbul:v2" and fix.new_model == DEFAULT_MODEL
    assert fix.old_voice == "anushka" and fix.new_voice == DEFAULT_SPEAKER
    assert not plan.manual_review


def test_null_tts_config_needs_no_db_change() -> None:
    # inherits the adapter default now that the code fix landed
    tenants = [_tenant("mgmstage", None)]
    plan = compute_plan(tenants)
    assert not plan.fixes
    assert not plan.manual_review
    assert any("mgmstage" in note for note in plan.unaffected)


def test_already_v3_valid_tenant_is_left_alone() -> None:
    tenants = [_tenant("acme-sports", {"provider": "sarvam", "model": "bulbul:v3",
                                        "voice_id": "ritu"})]
    plan = compute_plan(tenants)
    assert not plan.fixes
    assert not plan.manual_review


def test_unknown_invalid_voice_flagged_for_manual_review_not_guessed() -> None:
    # A v2-only speaker this script has no verified replacement for must be
    # reported, never silently mapped to a guessed speaker.
    tenants = [_tenant("weird", {"provider": "sarvam", "model": "bulbul:v3",
                                  "voice_id": "meera"})]
    plan = compute_plan(tenants)
    assert not plan.fixes
    assert len(plan.manual_review) == 1
    assert plan.manual_review[0].voice_id == "meera"


def test_non_sarvam_provider_is_out_of_scope() -> None:
    tenants = [_tenant("indicf5-tenant", {"provider": "indicf5", "voice_id": "indicf5"})]
    plan = compute_plan(tenants)
    assert not plan.fixes
    assert not plan.manual_review
    assert not plan.unaffected  # not even reported as "no change needed" — out of scope entirely


def test_model_only_pinned_without_voice_still_fixed() -> None:
    tenants = [_tenant("dev", {"provider": "sarvam", "model": "bulbul:v2"})]
    plan = compute_plan(tenants)
    assert len(plan.fixes) == 1
    assert plan.fixes[0].new_model == DEFAULT_MODEL
    assert plan.fixes[0].old_voice is None and plan.fixes[0].new_voice is None
