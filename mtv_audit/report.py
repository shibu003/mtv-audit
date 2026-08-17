"""Receipt renderer (Japanese-first, per GTM constraint §2).

Required fields (§6): totals / waste% by channel / top-10 wasteful loops
with excerpts / monthly recoverable under λ ∈ {saver, balanced, optimizer}
/ methodology notes / replay verification status.
"""
from __future__ import annotations

import datetime as _dt

from .attribution import ALL_CHANNELS, RECOVERABLE_CHANNELS, Ledger
from .model import Session
from .pricing import PriceBook
from .replay import ReplayResult

CHANNEL_JA = {
    "retry": "retry｜汚染再試行（失敗トレースの再混入）",
    "clean": "clean｜腐敗文脈の再読（無関連ブロックの再送）",
    "comm": "comm｜全状態再放送（サブエージェント差分同期の不在)",
    "deep": "deep｜過剰思考（自明ステップへのthinking）",
    "stop": "stop｜ブレーカー不在（完了後の継続消費）",
    "model": "model｜過剰段（最上位モデルの自明ステップ投入）※フラグのみ",
}


def _fmt_usd(x: float) -> str:
    return f"${x:,.4f}" if x < 1 else f"${x:,.2f}"


def _fmt_tok(x: float) -> str:
    return f"{int(round(x)):,}"


PRICE_TABLE_VERSION = "2026-06-11"  # bump when DEFAULT_PRICES changes


def render_receipt(session: Session, ledgers: dict[str, Ledger],
                   book: PriceBook, detail_dial: str = "balanced",
                   sessions_per_month: int = 100,
                   replay: ReplayResult | None = None,
                   data_provenance: str = "fixture") -> str:
    tot = session.total_reported()
    total_usd = sum(
        book.turn_cost_usd(t.model, t.usage) for t in session.assistant_turns()
    )
    detail = ledgers[detail_dial]
    ch = detail.channel_totals()

    lines: list[str] = []
    a = lines.append
    a("# MTV無駄監査 領収書（Waste Audit Receipt）")
    a("")
    a(f"- セッション: `{session.meta.get('session_id', 'unknown')}`  /  ソース: `{session.source_path}`")
    a(f"- 監査日: {_dt.date.today().isoformat()}  /  詳細台帳のダイヤル: **{detail_dial}**")
    a(f"- ターン数: {len(session.turns)}（うちアシスタント {len(session.assistant_turns())}）")
    a(f"- **data_provenance: {data_provenance}**  /  price_table_version: {PRICE_TABLE_VERSION}")
    a("")
    a("## 1. セッション総計")
    a("")
    a(f"処理トークン総量 **{_fmt_tok(tot['grand_total'])}**"
      f"（プロンプト側 {_fmt_tok(tot['prompt_total'])}"
      f" ＝ 通常入力 {_fmt_tok(tot['input_tokens'])}"
      f" ＋ キャッシュ読込 {_fmt_tok(tot['cache_read_input_tokens'])}"
      f" ＋ キャッシュ書込 {_fmt_tok(tot['cache_creation_input_tokens'])}"
      f"、出力側 {_fmt_tok(tot['output_tokens'])}）。"
      f"請求額換算 **{_fmt_usd(total_usd)}**（キャッシュ料率を反映したブレンド単価で算定）。")
    a("")
    a("## 2. チャネル別無駄台帳")
    a("")
    a(f"| チャネル | 無駄トークン | 金額 | 総額比 | 件数 |")
    a("|---|---:|---:|---:|---:|")
    for c in ALL_CHANNELS:
        row = ch[c]
        pct = (row["usd"] / total_usd * 100) if total_usd else 0.0
        flag = "（参考値）" if c == "model" else ""
        a(f"| {CHANNEL_JA[c]} | {_fmt_tok(row['tokens'])} | {_fmt_usd(row['usd'])} | {pct:.1f}%{flag} | {int(row['count'])} |")
    rec_usd = detail.recoverable_usd()
    rec_pct = (rec_usd / total_usd * 100) if total_usd else 0.0
    a("")
    a(f"**回収可能（{detail_dial}・リプレイ検証前の上限見込み）: {_fmt_usd(rec_usd)} ＝ 請求額の {rec_pct:.1f}%**")
    a("（model チャネルは下位モデルでのリプレイ確認まで合計に含めません。）")
    a("")
    a("## 3. 最も無駄なループ Top 10（抜粋付き）")
    a("")
    a("| # | チャネル | 対象ブロック | 再送回数 | ターン範囲 | トークン | 金額 | 抜粋 |")
    a("|---|---|---|---:|---|---:|---:|---|")
    for i, item in enumerate(detail.top_items(10), 1):
        ex = item["excerpt"].replace("|", "\\|")
        a(f"| {i} | {item['channel']} | `{item['block_id']}` | {item['repeat']} | {item['turn_span']} | "
          f"{_fmt_tok(item['tokens'])} | {_fmt_usd(item['usd'])} | {ex}… |")
    a("")
    a("## 4. 月次回収見込み（λダイヤル別）")
    a("")
    a(f"前提: 本セッションと同等の負荷 × **月 {sessions_per_month} セッション**（CLI引数 `--sessions-per-month` で変更可）。")
    a("")
    a("| λ（ユーザーダイヤル） | 1セッション回収額 | 請求額比 | 月次回収見込み |")
    a("|---|---:|---:|---:|")
    for name in ("saver", "balanced", "optimizer"):
        led = ledgers[name]
        usd = led.recoverable_usd()
        pct = (usd / total_usd * 100) if total_usd else 0.0
        a(f"| {name} | {_fmt_usd(usd)} | {pct:.1f}% | {_fmt_usd(usd * sessions_per_month)} |")
    a("")
    a("## 5. リプレイ検証ステータス")
    a("")
    if replay is None:
        a("- ステータス: **NOT_RUN**")
    else:
        a(f"- ステータス: **{replay.status}**")
        if replay.note:
            a(f"- 備考: {replay.note}")
    a("- 検証方法: 低MTV判定トークン（retry / clean）を除去した反実仮想セッションを再実行し、"
      "元セッションで成立していた成功条件（テスト合格等）が維持されることを確認する。"
      "検証完了後、本領収書の「上限見込み」は「検証済み削減額」に置換される。")
    a("")
    a("## 6. 手法注記（監査の前提）")
    a("")
    a("- 二重計上なし: 各（ターン, ブロック）のプロンプト側トークンは最大1チャネルにのみ帰属。"
      "優先順位は retry > clean > comm > deep > stop。model はフラグのみで合計外。")
    a("- トークン推定: ブロック単位は chars/4 の推定値を、各ターンのAPI報告 usage に対して"
      "スケーリング（turn_scale, 0.25–4.0 にクランプ）し、請求トークン相当で計上。")
    a("- 金額換算: 文脈側はそのターンの実キャッシュ構成を反映したブレンド単価、"
      "出力側は当該モデルの出力単価。単価表は設定ファイルで上書き可能（既定値は要照合: "
      "https://docs.claude.com の公式価格ページ）。")
    a("- 関連度スコア: v0 は字句ベース（目標キーワード命中率）。embedding ベースは"
      "実ログでの誤分類率を見て追加判断。")
    a("- comm: 単一セッションではサブエージェント呼び出しペイロードの重複率で近似。"
      "マルチエージェント構成では「総トークン − 有用出力 ＝ 管理コスト」枠組みに拡張予定。")
    a("- 本書の数値は監査対象ログのみから算出。業界横断の削減率を主張するものではない。")
    a("")
    a("---")
    a("*Generated by mtv-audit (Stage 1). 記録は追記専用・再現可能。*")
    return "\n".join(lines) + "\n"
