# Project instructions for Claude

## Model routing + task loop

For any substantive task (feature, bugfix, refactor — anything beyond a trivial
one-file tweak or a question), follow this loop:

1. **Plan — Fable (main session).** Do the planning, design, and task breakdown
   in the main loop; do not delegate planning to a cheaper model. Clarify scope
   with the user before fanning out if the request is ambiguous.
2. **Implement — Sonnet subagents.** Dispatch implementation work via the Agent
   tool with `model: "sonnet"`. Give each agent a self-contained brief: files,
   the plan step, repo conventions, and the test command it must run. Parallel
   agents only for independent steps.
3. **Test/Review — Opus subagents.** After implementation, dispatch a review
   agent with `model: "opus"` to (a) run the relevant test suite and (b) review
   the diff for correctness against the plan. It must report concrete findings
   with file:line, not general approval.
4. **Loop.** If the review finds real problems, feed the findings back into a
   new Sonnet fix round, then re-review with Opus. Repeat until the review is
   clean or 3 rounds have run — after 3 rounds, stop and surface the remaining
   findings to the user instead of iterating further.

The main (Fable) session stays the orchestrator throughout: it synthesizes
agent output, makes the calls between rounds, and writes the final summary.
It should not hand-write implementation code for work already delegated —
route fixes back through the loop.

Exempt from the loop: trivial mechanical edits, doc-only changes, config
tweaks, and investigation/debugging questions where the deliverable is an
assessment — handle those directly in the main session.

## Verification

Tests live in `tests/` (pytest, run via `.venv/bin/python -m pytest`). Two
known pre-existing failures on main (unrelated to most work):
`test_chat_routes.py::test_claim_session_and_agent_ws` and
`test_prompts.py::test_chatbot_prompt_has_scope_guardrails` — don't chase
these unless the task is about them.

## Existing standing preferences

- Never use "thesis"/academic framing in commit messages, PRs, or committed docs.
- Always confirm with the user before editing any prompt file (`prompts.py` etc.),
  even in auto mode.
- Commit only when asked; commits go directly to `main` per current workflow.
