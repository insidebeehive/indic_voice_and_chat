"""Group similar customer chat messages and show what they cost.

Input: the CSV written by scripts/sql/export_customer_queries.sql.
Usage:
    python3 scripts/analyze_customer_queries.py queries.csv [--top 40] [--threshold 0.55]
        [--in-rate 1.50 --cached-rate 0.15 --out-rate 9.00]   # USD per 1M tokens

Needs only the Python standard library. Nothing is sent anywhere; digits,
emails and phone numbers are masked in the samples it prints.
"""
import argparse
import csv
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict

csv.field_size_limit(10_000_000)

_EMAIL = re.compile(r"\S+@\S+")
_DIGITS = re.compile(r"\d+")
_PUNCT = re.compile(r"[^\w\s#]", re.UNICODE)
_SPACE = re.compile(r"\s+")


def normalise(text):
    t = unicodedata.normalize("NFKC", text or "").lower()
    t = _EMAIL.sub(" @ ", t)
    t = _DIGITS.sub("#", t)
    t = _PUNCT.sub(" ", t)
    return _SPACE.sub(" ", t).strip()


def mask(text, width=90):
    t = _EMAIL.sub("<email>", text or "")
    t = _DIGITS.sub("#", t)
    t = _SPACE.sub(" ", t).strip()
    return t if len(t) <= width else t[: width - 1] + "…"


def grams(norm):
    padded = f" {norm} "
    return Counter(padded[i:i + 3] for i in range(len(padded) - 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--threshold", type=float, default=0.55,
                    help="cosine similarity needed to join a group (0-1); lower = bigger groups")
    ap.add_argument("--in-rate", type=float, default=1.50)
    ap.add_argument("--cached-rate", type=float, default=0.15)
    ap.add_argument("--out-rate", type=float, default=9.00)
    a = ap.parse_args()

    def i(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return 0

    rows = []
    with open(a.csv, newline="", encoding="utf-8-sig") as f:
        # Skip anything psql printed ahead of the CSV header (e.g. "SET" when
        # run without -q).
        lines = iter(f)
        for line in lines:
            if line.startswith("session_id,"):
                break
        else:
            sys.exit("no CSV header (session_id,...) found -- is this the output of "
                     "scripts/sql/export_customer_queries.sql?")
        for r in csv.DictReader(lines, fieldnames=next(csv.reader([line]))):
            norm = normalise(r.get("content", ""))
            if not norm or norm in ("[audio]", "audio"):
                continue
            inp, cached, out = i(r.get("input_tokens")), i(r.get("cached_tokens")), i(r.get("output_tokens"))
            has_turn = bool(r.get("llm_calls"))
            cost = ((inp - cached) * a.in_rate + cached * a.cached_rate + out * a.out_rate) / 1e6
            rows.append({
                "norm": norm, "text": r.get("content", ""),
                "first": (r.get("first_msg") or "").lower() in ("t", "true", "1"),
                "has_turn": has_turn, "cost": cost if has_turn else 0.0,
                "calls": i(r.get("llm_calls")),
                "tool": r.get("first_tool") or ("(no tool)" if has_turn else "(no turn)"),
            })
    if not rows:
        sys.exit("no customer messages in the CSV")

    # Group identical normalised texts first, then cluster the unique texts
    # (most frequent first) by char-trigram TF-IDF cosine, using an inverted
    # index so each text is only compared with group leaders it shares grams with.
    by_norm = defaultdict(list)
    for r in rows:
        by_norm[r["norm"]].append(r)
    uniq = sorted(by_norm, key=lambda n: -len(by_norm[n]))
    df = Counter()
    for n in uniq:
        df.update(set(grams(n)))
    N = len(uniq)

    def vec(n):
        g = grams(n)
        v = {k: c * math.log((N + 1) / (df[k] + 1) + 1) for k, c in g.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {k: x / norm for k, x in v.items()}

    leaders, index, members = [], defaultdict(list), []
    for n in uniq:
        v = vec(n)
        scores = defaultdict(float)
        for k, x in v.items():
            for li in index[k]:
                scores[li] += x * leaders[li][k]
        best = max(scores.items(), key=lambda kv: kv[1], default=(None, 0.0))
        if best[0] is not None and best[1] >= a.threshold:
            members[best[0]].append(n)
        else:
            li = len(leaders)
            leaders.append(v)
            members.append([n])
            for k in v:
                index[k].append(li)

    total_msgs = len(rows)
    total_cost = sum(r["cost"] for r in rows)
    turns = [r for r in rows if r["has_turn"]]
    no_tool = [r for r in turns if r["tool"] == "(no tool)"]
    kb_only = [r for r in turns if r["tool"] == "search_knowledge_base"]

    print(f"Customer messages: {total_msgs}  (first message of a session: {sum(r['first'] for r in rows)})")
    print(f"Matched to an agent turn: {len(turns)}  |  attributed LLM cost: ${total_cost:.4f}"
          f"  |  avg per turn: ${total_cost / max(len(turns), 1):.5f}")
    for label, sub in (("no tool called", no_tool), ("knowledge base only (first tool)", kb_only)):
        c = sum(r["cost"] for r in sub)
        print(f"  {label:34s} {len(sub):6d} turns ({100 * len(sub) / max(len(turns), 1):4.1f}%)"
              f"  cost share {100 * c / max(total_cost, 1e-12):4.1f}%")
    print(f"Groups found: {len(members)} (threshold {a.threshold})\n")

    groups = [[r for n in m for r in by_norm[n]] for m in members]
    groups.sort(key=len, reverse=True)

    print(f"Top {a.top} groups by message count")
    print("-" * 100)
    cum = 0
    for gi, rs in enumerate(groups[: a.top], 1):
        cum += len(rs)
        gt = [r for r in rs if r["has_turn"]]
        cost = sum(r["cost"] for r in rs)
        tools = Counter(r["tool"] for r in gt).most_common(3)
        samples, seen = [], set()
        for r in sorted(rs, key=lambda r: -len(by_norm[r["norm"]])):
            if r["norm"] not in seen:
                seen.add(r["norm"])
                samples.append(mask(r["text"]))
            if len(samples) == 3:
                break
        print(f"#{gi:<3d} {len(rs):6d} msgs  {100 * len(rs) / total_msgs:5.1f}%  (cum {100 * cum / total_msgs:5.1f}%)"
              f"  first-msg {100 * sum(r['first'] for r in rs) / len(rs):3.0f}%"
              f"  cost ${cost:.4f} ({100 * cost / max(total_cost, 1e-12):4.1f}%)"
              f"  avg calls {sum(r['calls'] for r in gt) / max(len(gt), 1):.2f}")
        print("       tools: " + ", ".join(f"{t} {100 * c / max(len(gt), 1):.0f}%" for t, c in tools))
        for s in samples:
            print(f"       · {s}")
    singles = sum(1 for g in groups if len(g) == 1)
    print("-" * 100)
    print(f"Groups of size 1 (one-off messages): {singles} ({100 * singles / total_msgs:.1f}% of messages)")


if __name__ == "__main__":
    main()
