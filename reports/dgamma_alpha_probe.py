"""D-γ α-divergence probe (prediction-based, $0 offline).

Hypothesis (a): the seg-1 vs seg-2 α gap (2.524 vs 0.816) is a chars/4
composition bias — chars/4 is tuned for English ASCII (~4 chars/token) but
undercounts CJK (~1.5 chars/token). PRE-REGISTERED weights (literature, NOT
fit to this session): ASCII = 0.25 tok/char, CJK = 0.667 tok/char.

Predictions (set before measuring):
  1. seg-1 CJK fraction > seg-2 CJK fraction.
  2. Recomputing content est with the composition model and refitting α per
     segment converges BOTH α -> ~1.0 with R² maintained (>0.9).
Reject (a) if α does not converge or R² breaks (= backward curve-fitting).
"""
import sys

from mtv_audit.parser import load_session

CJK_RANGES = [
    (0x3000, 0x303F), (0x3040, 0x309F), (0x30A0, 0x30FF),  # CJK punct, hira, kata
    (0x4E00, 0x9FFF), (0x3400, 0x4DBF),                    # CJK unified (+ext A)
    (0xFF00, 0xFFEF),                                       # fullwidth forms
]
ASCII_TOK_PER_CHAR = 0.25      # chars/4 baseline (English-tuned)
CJK_TOK_PER_CHAR = 0.667       # ~1.5 chars/token, pre-registered


def is_cjk(ch: str) -> bool:
    o = ord(ch)
    return any(lo <= o <= hi for lo, hi in CJK_RANGES)


def comp_est(text: str) -> tuple[float, int, int]:
    """(composition-aware tokens, cjk_chars, total_chars)."""
    cjk = sum(1 for ch in text if is_cjk(ch))
    total = len(text)
    tok = ASCII_TOK_PER_CHAR * (total - cjk) + CJK_TOK_PER_CHAR * cjk
    return tok, cjk, total


def ols(rows):
    n = len(rows)
    if n < 2:
        return (0.0, 1.0, 0.0)
    sx = sum(c for _, c in rows); sy = sum(p for p, _ in rows)
    sxx = sum(c * c for _, c in rows); sxy = sum(p * c for p, c in rows)
    denom = n * sxx - sx * sx
    if denom == 0:
        return (sy / n, 1.0, 0.0)
    a = (n * sxy - sx * sy) / denom
    o = (sy - a * sx) / n
    mean = sy / n
    ss_tot = sum((p - mean) ** 2 for p, _ in rows) or 1.0
    ss_res = sum((p - (o + a * c)) ** 2 for p, c in rows)
    return (o, a, 1 - ss_res / ss_tot)


def run(path):
    s = load_session(path)
    # walk turns, split at compaction boundaries, accumulate BOTH estimators.
    segs = []  # each: {rows_old, rows_new, cjk, total}
    cur = None

    def newseg(start):
        return {"start": start, "rows_old": [], "rows_new": [],
                "cjk": 0, "total": 0, "cum_old": 0.0, "cum_new": 0.0}

    cur = newseg(0)
    for t in s.turns:
        if t.is_compact_summary:
            if cur["rows_old"]:
                segs.append(cur)
            cur = newseg(t.index)
            for b in t.blocks:
                tok, cjk, tot = comp_est(b.text)
                cur["cum_old"] += b.est_tokens
                cur["cum_new"] += tok
                cur["cjk"] += cjk; cur["total"] += tot
            continue
        if t.role == "assistant" and t.usage:
            P = t.reported_prompt_tokens()
            if P > 0:
                cur["rows_old"].append((float(P), cur["cum_old"]))
                cur["rows_new"].append((float(P), cur["cum_new"]))
        for b in t.blocks:
            tok, cjk, tot = comp_est(b.text)
            cur["cum_old"] += b.est_tokens
            cur["cum_new"] += tok
            cur["cjk"] += cjk; cur["total"] += tot
    if cur["rows_old"]:
        segs.append(cur)

    print(f"session: {path.split('/')[-1]}")
    print(f"{'seg_start':>9} {'n':>5} {'CJKfrac':>8} "
          f"{'alpha_old':>10} {'R2_old':>8} {'alpha_new':>10} {'R2_new':>8}")
    for sg in segs:
        o1, a1, r1 = ols(sg["rows_old"])
        o2, a2, r2 = ols(sg["rows_new"])
        cjkf = sg["cjk"] / sg["total"] if sg["total"] else 0.0
        print(f"{sg['start']:>9} {len(sg['rows_old']):>5} {cjkf:>8.3f} "
              f"{a1:>10.3f} {r1:>8.4f} {a2:>10.3f} {r2:>8.4f}")


if __name__ == "__main__":
    run(sys.argv[1])
