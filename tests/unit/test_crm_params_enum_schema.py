import re

from src.bootstrap import _crm_params_to_schema
from src.chatbot.catalog import PLAYER_TOOLS
from src.dialogue.packs.betting import COMMON_LOOKUPS_BLOCK


def test_transactions_type_optional_with_enum():
    schema = _crm_params_to_schema(
        PLAYER_TOOLS["get_player_transactions"]["parameters"])
    assert "type" not in schema.get("required", [])
    assert "limit" in schema["required"]
    assert schema["properties"]["type"]["enum"] == [
        "deposit", "withdrawal", "casino", "sports"]


def test_params_without_enum_unchanged():
    schema = _crm_params_to_schema({
        "q": {"type": "string", "source": "llm", "description": "d"}})
    assert schema == {"type": "object",
                      "properties": {"q": {"type": "string", "description": "d"}},
                      "required": ["q"]}


def test_betting_lookups_use_withdrawal():
    assert "type withdrawal)" in COMMON_LOOKUPS_BLOCK
    # "withdraw" is not an API value; it must not appear as a word anywhere
    # in the lookup rules the model picks the type from.
    assert not re.search(r"\bwithdraw\b", COMMON_LOOKUPS_BLOCK)
