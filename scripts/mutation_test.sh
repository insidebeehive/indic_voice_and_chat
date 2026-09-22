#!/usr/bin/env bash
# Mutation testing, scoped to the modules where an unasserted behaviour costs
# something.
#
# Why scoped: mutmut runs the test suite once per mutant. src/ is ~46k lines,
# which is order 10-20k mutants; at the full suite's ~140s that is weeks of
# compute. Each target below is paired with only the tests that exercise it,
# which brings a module's run down to minutes.
#
# Why these modules: they are the ones where a test that passes without
# asserting anything has a real cost -- customer-facing guards, PII redaction,
# vendor signing, and billing arithmetic. Line coverage cannot distinguish
# "this line ran" from "this line's behaviour was checked", and every vacuous
# test found in this repo so far had full line coverage.
#
#   ./scripts/mutation_test.sh                 # list targets
#   ./scripts/mutation_test.sh guards          # run one target
#   ./scripts/mutation_test.sh --all           # every target, sequentially
#
# After a run:
#   .venv/bin/mutmut results                   # survivors
#   .venv/bin/mutmut show <id>                 # the diff of one survivor
#   .venv/bin/mutmut html                      # browsable report
#
# Reading the output: a SURVIVOR is a question, not a defect. Mutating a log
# message, an unreachable defensive branch, or a `or 0.0` fallback survives
# harmlessly. A survivor in redaction, a guard, or the signing path is worth
# reading properly -- that is the whole reason for the target list.
set -euo pipefail

cd "$(dirname "$0")/.."
PY=.venv/bin/python
MUTMUT=.venv/bin/mutmut

if [[ ! -x "$MUTMUT" ]]; then
    echo "mutmut not installed. Run: .venv/bin/pip install -e '.[dev]'" >&2
    exit 1
fi

# target key -> "source path :: test paths"
targets() {
    cat <<'EOF'
guards|src/rag/context_builder.py|tests/unit/test_rag_context.py tests/unit/test_voicebot_sentence_guard.py tests/unit/test_chatbot_previous_conversation.py
redaction|src/chatbot/tool_executor.py|tests/unit/test_chat_tool_executor.py tests/unit/test_chat_tools_resolved_endpoint.py
parser|src/dialogue/response_parser.py|tests/unit/test_response_parser.py
cost|src/api/chat_cost.py|tests/unit/test_chat_cost.py tests/unit/test_catalog_routes.py
deposit|src/chatbot/deposit_verification.py|tests/unit/test_deposit_ticket_outbound.py tests/unit/test_deposit_verification_executor.py tests/unit/test_deposit_verification_timeout.py
gemini|src/providers/llm/gemini.py|tests/unit/test_gemini_llm_adapter.py
EOF
}

run_target() {
    local key="$1"
    local line src tests
    line="$(targets | grep "^${key}|" || true)"
    if [[ -z "$line" ]]; then
        echo "unknown target: ${key}" >&2
        exit 1
    fi
    src="$(echo "$line" | cut -d'|' -f2)"
    tests="$(echo "$line" | cut -d'|' -f3)"

    echo "=== ${key}: ${src}"
    echo "    tests: ${tests}"
    # --CI so a surviving mutant is reported rather than failing the script:
    # survivors are the output of interest here, not an error condition.
    "$MUTMUT" run \
        --paths-to-mutate "$src" \
        --tests-dir tests/ \
        --runner "$PY -m pytest -x -q ${tests}" \
        --no-progress \
        --CI || true
    echo
    "$MUTMUT" results
}

if [[ $# -eq 0 ]]; then
    echo "targets:"
    targets | while IFS='|' read -r key src _; do printf "  %-10s %s\n" "$key" "$src"; done
    echo
    echo "usage: $0 <target> | --all"
    exit 0
fi

if [[ "$1" == "--all" ]]; then
    targets | cut -d'|' -f1 | while read -r key; do run_target "$key"; done
else
    run_target "$1"
fi
