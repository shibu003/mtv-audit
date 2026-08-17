"""D-γ FINAL measurement: real tokenizer (tiktoken o200k proxy for Anthropic).

With a real tokenizer the per-block billed size is just tokenizer(text); no α
is needed (α·est was only estimating this). This sidesteps the unexplained α.

Sanity gate (run FIRST, per founder): does the real-token DENSITY (real/est)
per kind/segment match the earlier observation? The seg-1 α was 2.524 — if the
global density ≈ 2.5 then α is genuine token density (-> RED holds); if density
≈ 1.0-1.3 while α=2.5, the excess is NOT density (truncation/residency/fit) and
the per-block billing at 2.5 was an over-bill (-> re-score likely).

Then: recompute the H23 net delta with real tokens and read the sign.
"""
import sys

import tiktoken

from mtv_audit.parser import load_session
from mtv_audit.svm import (
    CALL_OVERHEAD, P_READ, P_WRITE, build_block_lives, calibrate,
)

ENC = tiktoken.get_encoding("o200k_base")


def rtok(text: str) -> int:
    return len(ENC.encode(text or "", disallowed_special=()))


def run(path: str):
    s = load_session(path)
    B = next(t.index for t in s.turns if t.is_compact_summary)
    cal = calibrate(s)
    seg_alpha = {sg.start_turn: sg.scale for sg in cal.segments}

    # ---- SANITY: per-kind / per-segment real-token density (real/est) ----
    print("== SANITY: real-token density (Σreal_tok / Σest_tok) ==")
    print(f"{'segment':>10} {'kind':>12} {'est':>10} {'real':>10} {'density':>8}")
    for label, lo, hi, a in [("seg1", 0, B, seg_alpha.get(0)),
                             ("seg2", B, 10**9, seg_alpha.get(B))]:
        per_kind = {}
        for t in s.turns:
            if lo <= t.index < hi and not t.is_compact_summary:
                for b in t.blocks:
                    if b.kind in ("tool_result", "tool_use", "text", "thinking"):
                        e, r = per_kind.get(b.kind, (0, 0))
                        per_kind[b.kind] = (e + b.est_tokens, r + rtok(b.text))
        tot_e = sum(e for e, _ in per_kind.values())
        tot_r = sum(r for _, r in per_kind.values())
        for k, (e, r) in sorted(per_kind.items(), key=lambda x: -x[1][0]):
            print(f"{label:>10} {k:>12} {e:>10,} {r:>10,} {r/e if e else 0:>8.3f}")
        dens = tot_r / tot_e if tot_e else 0
        print(f"{label:>10} {'ALL':>12} {tot_e:>10,} {tot_r:>10,} {dens:>8.3f}"
              f"   <-- compare to α={a:.3f}")
        print()

    # ---- VERDICT: H23 net delta with REAL tokens (no α) ----
    lives = build_block_lives(s, cal)
    assist_after = [t.index for t in s.assistant_turns() if t.index > B]
    total_billed_real = sum(rtok(bl.text) for bl in lives)

    net = 0.0
    gross = 0.0
    n = 0
    rows = []
    for bl in lives:
        if bl.birth_turn >= B:
            continue
        refs_after = [r for r in bl.ref_turns if r > B]
        if not refs_after:
            continue
        billed = rtok(bl.text)               # real token billed size
        first = min(refs_after)
        resid = sum(1 for i in assist_after if i <= first)
        reread = billed * P_WRITE + CALL_OVERHEAD
        rent = billed * P_READ * resid
        nb = reread - rent
        net += nb
        gross += billed
        n += 1
        rows.append((billed, resid, nb, bl.kind, bl.tool_name, bl.excerpt))

    print("== VERDICT: H23 with real tokens (no α) ==")
    print(f"disagreement blocks: {n}")
    print(f"total pageable billed (real tok): {total_billed_real:,.0f}")
    print(f"gross disagreement mass (real tok): {gross:,.0f}"
          f"  ({gross/total_billed_real:.2%} of billed)")
    print(f"NET delta (reread_avoided - extra_rent): {net:,.0f}"
          f"  ({net/total_billed_real:+.2%} of billed)")
    print(f"sign: {'POSITIVE -> re-score' if net > 0 else 'NEGATIVE -> RED'}"
          f"   (θ_δ=10%)")
    print("\ntop disagreement blocks by real billed mass:")
    for billed, resid, nb, kind, tn, ex in sorted(rows, reverse=True)[:10]:
        print(f"  [{tn or kind}] real_tok={billed} resid_after={resid} net={nb:,.0f}")
        print(f"      «{ex}»")


if __name__ == "__main__":
    run(sys.argv[1])
