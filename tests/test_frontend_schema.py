"""Schema-coupling guard between the leaderboard frontend and aggregate_results.

frontend/index.html is a dependency-free static page that reads
``data/results/leaderboard.json`` (the file aggregate_results.py writes). It has
no build step and no shared type with the Python side, so the only thing keeping
the two in sync is that every key the JS dereferences is actually emitted by
``compute_leaderboard``. These tests pin that contract:

1. Every top-level and per-model key the frontend reads is a subset of what
   ``compute_leaderboard`` emits (so a rename/removal on the Python side that
   would silently break the page fails CI here instead).
2. The new fields surfaced by the frontend (holdout gap, pass^k, premature-end,
   the efficacy CI whisker, the Pareto/value columns) are present in the emitted
   entry with the expected types.
3. The page's render() runs against synthetic populated / empty / degraded JSON
   under a headless DOM (node + a tiny DOM stub), so the actual render path — not
   just the key list — is exercised for all three data states. These tests SKIP
   when node is unavailable (e.g. minimal CI images) rather than failing, since
   the page ships dependency-free and CI can't render a browser.

The "expected keys" lists below are the ground truth for what index.html touches;
update them in lockstep when the page starts reading a new field.
"""

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.aggregate_results import compute_leaderboard

FRONTEND = Path(__file__).resolve().parents[1] / "frontend" / "index.html"

# Keys index.html dereferences off the top-level leaderboard object. Kept as the
# authoritative list of the page's coupling to the aggregate output.
FRONTEND_TOPLEVEL_KEYS = {
    "updated",
    "models",
    "statistical_note",
    "holdout",
    # Visual overhaul (#56): version pill, domain filter source, per-domain
    # overlay for the domain view, and the rank-band count in the run summary.
    "version",
    "domains",
    "domain_scores",
    "n_rank_bands",
    # Corpus-health headline appended to the stat note (#71). Conditionally
    # emitted, but always present on a frame with model/scenario/efficacy data.
    "corpus_health",
    # Diagnostics panel: failure-mode breakdown reads the taxonomy header for
    # its stable column order (#55/#66).
    "failure_taxonomy",
    # Diagnostics panel: persona-stratified robustness table (#59/#70 H4).
    # Conditionally emitted — only when the run has non-cooperative rows — so
    # the subset test below uses the stratified fixture.
    "sim_profile_robustness",
}

# Keys index.html dereferences off each entry in leaderboard.models.
FRONTEND_MODEL_KEYS = {
    "name",
    "clear_score",
    "clear_score_ci",
    "efficacy",
    "task_completion",
    "tool_selection",
    "cost_per_task_usd",
    "reliability",
    "avg_latency_ms",
    "judge_agreement",
    "rank_band",
    # New fields surfaced by the holdout column + reliability detail line.
    "holdout_gap",
    "holdout_score",
    "premature_end_rate",
    "reliability_pass_hat_k",
    # Visual overhaul (#56): inline efficacy CI whisker reads efficacy_ci.
    "efficacy_ci",
    # Macro-averaged efficacy sub-readout + failure-profile card line (#55).
    "efficacy_macro_category",
    "efficacy_macro_category_ci",
    "failure_profile",
    # Consistency band (solid / avg / best, #71) in the reliability detail line.
    # Conditionally emitted: present whenever the parquet has reliability repeats.
    "consistency_band",
}


def _build_df(with_holdout=True, with_premature=True, with_pass_hat=True, with_sim_profiles=False):
    """A realistic results frame exercising every column the leaderboard reads.

    ``with_sim_profiles`` adds non-cooperative user-sim rows (issue #59) so the
    aggregate emits ``sim_profile_robustness``; cooperative rows carry no
    ``sim_profile`` value and count as cooperative via the fillna default.
    """
    rng = np.random.default_rng(11)
    judges = ["Kimi K2.6", "GLM-4.6", "Claude Opus 4.6"]
    models = {"GPT-5.5": 0.88, "Claude Sonnet 4.6": 0.78, "GPT-4.1 (anchor)": 0.55}
    halves = [(False, 0.0, "pub")]
    if with_holdout:
        halves.append((True, -0.15, "hold"))
    rows = []
    for holdout, adj, prefix in halves:
        for model, base in models.items():
            for s in range(32):
                for r in range(3):
                    eff = float(np.clip(base + adj + rng.normal(0, 0.03), 0, 1))
                    row = {
                        "scenario_id": f"{prefix}_{s:02d}",
                        "domain": "banking" if s % 2 else "customer_success",
                        "category": "adaptive_tool_use",
                        "model": model,
                        "holdout": holdout,
                        "efficacy": eff,
                        "task_completion": eff,
                        "tool_selection": eff,
                        "state_score": eff,
                        "cost_usd": 0.01 * (1 + list(models).index(model)),
                        "latency_ms": 2000.0,
                        "total_turns": 5,
                        "output_tokens": 500 + r * 50,
                        "reliability_pass_rate": base,
                        "reliability_consistency": 0.9,
                        "tc_agreement": 0.9,
                        "ts_agreement": 0.9,
                    }
                    if with_premature:
                        row["premature_end"] = r == 0 and model == "GPT-4.1 (anchor)"
                    # Failure-mode column (#55): below-threshold rows fail.
                    row["failure_mode"] = None if eff >= 0.7 else "incomplete-task"
                    if with_pass_hat:
                        row["reliability_pass_hat_1"] = base
                        row["reliability_pass_hat_2"] = base * 0.9
                        row["reliability_pass_hat_3"] = base * 0.8
                    for j in judges:
                        row[f"tc_{j}"] = float(np.clip(eff + rng.normal(0, 0.05), 0, 1))
                        row[f"ts_{j}"] = float(np.clip(eff + rng.normal(0, 0.05), 0, 1))
                    rows.append(row)
                    # Non-cooperative behavioral-profile rows (issue #59): public
                    # half only, lower efficacy. Excluded from public aggregates;
                    # surfaced via sim_profile_robustness.
                    if with_sim_profiles and not holdout and r == 0 and s < 8:
                        for profile in ("impatient", "technically-confused"):
                            nrow = dict(row)
                            nrow["sim_profile"] = profile
                            nrow["efficacy"] = float(max(0.0, eff - 0.2))
                            rows.append(nrow)
    return pd.DataFrame(rows)


class TestFrontendSchemaCoupling:
    def test_toplevel_keys_are_subset_of_emitted(self):
        # Stratified fixture: sim_profile_robustness is only emitted when the
        # run has non-cooperative rows, and the frontend reads it when present.
        lb = compute_leaderboard(_build_df(with_sim_profiles=True))
        missing = FRONTEND_TOPLEVEL_KEYS - set(lb.keys())
        assert not missing, f"frontend reads top-level keys aggregate no longer emits: {missing}"

    def test_model_keys_are_subset_of_emitted(self):
        lb = compute_leaderboard(_build_df())
        assert lb["models"], "fixture produced no models"
        emitted = set(lb["models"][0].keys())
        missing = FRONTEND_MODEL_KEYS - emitted
        assert not missing, f"frontend reads model keys aggregate no longer emits: {missing}"

    def test_new_surfaced_fields_present_and_typed(self):
        lb = compute_leaderboard(_build_df())
        assert lb["holdout"]["present"] is True
        entry = lb["models"][0]
        # Holdout gap/score are floats here (this model has both halves).
        assert isinstance(entry["holdout_gap"], float)
        assert isinstance(entry["holdout_score"], float)
        # premature_end_rate is a float (a rate), pass^k is a dict keyed by k.
        assert isinstance(entry["premature_end_rate"], float)
        assert isinstance(entry["reliability_pass_hat_k"], dict)
        assert entry["reliability_pass_hat_k"], "expected per-k pass^k values"
        # Macro efficacy + failure profile (#55): float, [lo, hi], and a profile
        # dict with the full mode vocabulary.
        assert isinstance(entry["efficacy_macro_category"], float)
        ci = entry["efficacy_macro_category_ci"]
        assert isinstance(ci, list) and len(ci) == 2 and all(isinstance(v, float) for v in ci)
        assert isinstance(entry["failure_profile"], dict)
        assert set(entry["failure_profile"]) == {"n_rows", "n_failures", "failure_rate", "modes"}
        # Consistency band + corpus health (#71): present on this 3-run frame.
        band = entry["consistency_band"]
        assert isinstance(band, dict)
        assert band["solid_rate"] <= band["avg_pass_rate"] <= band["best_of_rate"]
        assert isinstance(lb["corpus_health"]["headline"], str)

    def test_sim_profile_robustness_emitted_only_when_stratified(self):
        # Stratified run: the table is present with the shape the diagnostics
        # panel renders (per-model per-profile pass rates + delta_vs_cooperative).
        lb = compute_leaderboard(_build_df(with_sim_profiles=True))
        spr = lb["sim_profile_robustness"]
        assert spr["cooperative_profile"] == "cooperative"
        assert spr["models"], "expected per-model robustness entries"
        profiles = next(iter(spr["models"].values()))
        assert {"cooperative", "impatient", "technically-confused"} <= set(profiles)
        assert profiles["cooperative"]["delta_vs_cooperative"] is None
        for name in ("impatient", "technically-confused"):
            entry = profiles[name]
            assert {
                "pass_rate",
                "mean_efficacy",
                "n_rows",
                "n_scenarios",
                "delta_vs_cooperative",
            } <= set(entry)
            assert isinstance(entry["delta_vs_cooperative"], float)
        # Cooperative-only run: the key is absent (no empty surface), and the
        # frontend hides the robustness section.
        assert "sim_profile_robustness" not in compute_leaderboard(_build_df())

    def test_new_fields_degrade_to_none_on_legacy_data(self):
        # No holdout / premature / pass^k columns (legacy parquet). The frontend
        # treats null as "render nothing extra"; aggregate must emit the keys with
        # null/empty so the JS `?.`/`!= null` guards have something to read.
        lb = compute_leaderboard(
            _build_df(with_holdout=False, with_premature=False, with_pass_hat=False)
        )
        assert lb["holdout"]["present"] is False
        entry = lb["models"][0]
        assert entry["holdout_gap"] is None
        assert entry["holdout_score"] is None
        assert entry["premature_end_rate"] is None
        assert entry["reliability_pass_hat_k"] == {}


class TestFrontendReadsDocumentedKeys:
    """Cheap drift guard: the keys we claim the frontend reads actually appear in
    index.html. Catches the inverse mistake — pruning a field from the page (so it
    silently stops surfacing) without updating this test's key list."""

    def test_declared_model_keys_appear_in_html(self):
        html = FRONTEND.read_text(encoding="utf-8")
        # Strip the demo-data block: it references model fields too, but we want to
        # confirm the *render* path reads each declared key, not the demo literal.
        html_no_demo = re.sub(r"function renderDemo\(\).*?^\s{8}\}", "", html, flags=re.S | re.M)
        for key in FRONTEND_MODEL_KEYS:
            assert key in html_no_demo, f"declared frontend key '{key}' not found in index.html"


# --- Headless render test (#56) -----------------------------------------------
# The page ships dependency-free; there is no bundler and CI has no browser. To
# still exercise render()/renderScatter()/computeFrontier()/valuePerDollar()
# against real aggregate output we extract the page's <script> body and run it in
# node under a minimal DOM stub. This catches a class of bugs the key-subset
# checks can't: a render path that reads an emitted key but mis-handles a null,
# or that throws on the empty / degraded shapes. Skips cleanly without node.

NODE = shutil.which("node")
_skip_no_node = pytest.mark.skipif(
    NODE is None, reason="node not available; headless render test skipped"
)

# A tiny DOM stub: enough surface for the page's render functions (getElementById,
# querySelector(All), createElement(NS), classList, style, innerHTML, dataset,
# addEventListener). It records innerHTML writes so the test can assert on output.
_DOM_STUB = r"""
class ClassList {
  constructor(){ this._s = new Set(); }
  add(...c){ c.forEach(x=>this._s.add(x)); }
  remove(...c){ c.forEach(x=>this._s.delete(x)); }
  toggle(c,on){
    if(on===undefined) on=!this._s.has(c);
    on?this._s.add(c):this._s.delete(c); return on;
  }
  contains(c){ return this._s.has(c); }
}
class El {
  constructor(tag){
    this.tagName = tag || 'DIV';
    this.children = []; this.childNodes = this.children;
    this.classList = new ClassList(); this.style = {}; this.dataset = {};
    this.attrs = {}; this._innerHTML = ''; this.textContent = '';
    this.nodeType = 1;
  }
  set innerHTML(v){ this._innerHTML = String(v); }
  get innerHTML(){ return this._innerHTML; }
  setAttribute(k,v){
    this.attrs[k]=String(v);
    if(k==='class'){
      this.classList=new ClassList();
      String(v).split(/\s+/).forEach(c=>c&&this.classList.add(c));
    }
  }
  getAttribute(k){ return this.attrs[k]; }
  appendChild(c){ this.children.push(c); return c; }
  removeChild(c){ const i=this.children.indexOf(c); if(i>=0) this.children.splice(i,1); return c; }
  addEventListener(){}
  querySelector(){ return null; }
  querySelectorAll(){ return []; }
}
const registry = {};
function getEl(id){ if(!registry[id]) registry[id]=new El('div'); return registry[id]; }
const ids = ['last-updated','version-pill','empty-state','results-view','domain-filter',
  'leaderboard-body','cards-view','holdout-th','stat-note','stat-note-mobile',
  'scatter-svg','scatter-panel','changelog-body','sc-tooltip',
  'diagnostics-panel','diag-failure','failure-modes-head','failure-modes-body',
  'diag-robustness','robustness-head','robustness-body',
  'diag-corpus','corpus-health-body'];
ids.forEach(getEl);
// scatter-svg needs a <title> child so the "keep title" clear logic has something.
const titleEl = new El('title'); getEl('scatter-svg').appendChild(titleEl);

global.document = {
  getElementById: getEl,
  querySelector(sel){ if(sel==='table tbody') return getEl('leaderboard-body'); return null; },
  querySelectorAll(){ return []; },
  createElement: tag => new El(tag),
  createElementNS: (ns,tag) => new El(tag),
};
global.location = { search: '' };
global.URLSearchParams = class { constructor(){} get(){ return null; } };
global.fetch = async () => { throw new Error('no network in test'); };
global.module = { exports: {} };
global.window = global;
"""

_RUNNER_TMPL = r"""
%(stub)s
%(script)s
// --- drive the render paths for the supplied state ---
const STATE = %(state)s;
const out = { ok: true, steps: [] };
function collectDiagnostics(out) {
  out.diag_panel_display = document.getElementById('diagnostics-panel').style.display;
  out.diag_failure_display = document.getElementById('diag-failure').style.display;
  out.diag_robustness_display = document.getElementById('diag-robustness').style.display;
  out.diag_corpus_display = document.getElementById('diag-corpus').style.display;
  out.diag_html = [
    document.getElementById('failure-modes-head').innerHTML,
    document.getElementById('failure-modes-body').innerHTML,
    document.getElementById('robustness-head').innerHTML,
    document.getElementById('robustness-body').innerHTML,
    document.getElementById('corpus-health-body').innerHTML,
  ].join('\n');
}
try {
  if (STATE === 'empty') {
    leaderboardData = { models: [] };
    onDataLoaded();
    out.steps.push('empty:' + (document.getElementById('empty-state').style.display === ''));
    out.empty_shown = document.getElementById('empty-state').style.display === '';
    out.results_hidden = document.getElementById('results-view').style.display === 'none';
  } else {
    if (STATE === 'demo') {
      renderDemo();
    } else {
      leaderboardData = %(data)s;
      onDataLoaded();
    }
    out.body_html = document.getElementById('leaderboard-body').innerHTML;
    out.cards_html = document.getElementById('cards-view').innerHTML;
    out.svg_children = document.getElementById('scatter-svg').children.length;
    out.holdout_th_display = document.getElementById('holdout-th').style.display;
    collectDiagnostics(out);
    // exercise a re-sort to make sure sortBy + render don't throw
    sortBy('value_per_dollar');
    sortBy('cost_per_task_usd');
    out.body_after_sort_len = document.getElementById('leaderboard-body').innerHTML.length;
    // pure-helper sanity
    const fr = computeFrontier(leaderboardData.models);
    out.frontier = [...fr];
    out.values = leaderboardData.models.map(m => valuePerDollar(m));
  }
} catch (e) {
  out.ok = false; out.error = String(e && e.stack || e);
}
console.log(JSON.stringify(out));
"""


def _extract_script(html: str) -> str:
    """Pull the page's inline <script> body, minus the loadData() bootstrap call.

    We drive the render functions directly from the test, so we strip the
    auto-run ``loadData();`` line (it would try to fetch over the network).
    """
    m = re.search(r"<script>(.*)</script>", html, flags=re.S)
    assert m, "no <script> block found in index.html"
    body = m.group(1)
    body = body.replace("loadData();", "// loadData(); (driven by test)")
    return body


def _run_node(state: str, data: dict | None) -> dict:
    html = FRONTEND.read_text(encoding="utf-8")
    runner = _RUNNER_TMPL % {
        "stub": _DOM_STUB,
        "script": _extract_script(html),
        "state": json.dumps(state),
        "data": json.dumps(data) if data is not None else "null",
    }
    # Write the runner to a temp file rather than passing via -e: the assembled
    # script (DOM stub + the whole page <script> + JSON payload) easily exceeds
    # the Windows command-line length limit.
    with tempfile.NamedTemporaryFile("w", suffix=".cjs", delete=False, encoding="utf-8") as fh:
        fh.write(runner)
        runner_path = fh.name
    try:
        proc = subprocess.run(
            [NODE, runner_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
    finally:
        Path(runner_path).unlink(missing_ok=True)
    assert proc.returncode == 0, f"node exited {proc.returncode}: {proc.stderr}\n{proc.stdout}"
    # The page logs one JSON line; take the last non-empty line.
    line = [ln for ln in proc.stdout.splitlines() if ln.strip()][-1]
    return json.loads(line)


def _populated_payload(degraded: bool = False) -> dict:
    """Real aggregate output, optionally stripped to the degraded (null) shape."""
    lb = compute_leaderboard(_build_df(with_holdout=not degraded, with_sim_profiles=not degraded))
    if degraded:
        # Degraded data: nulls for CI + holdout, empty pass^k. Mimics a run where
        # bootstrap/holdout didn't produce values. The page must still render.
        for mdl in lb["models"]:
            mdl["clear_score_ci"] = [None, None]
            mdl["efficacy_ci"] = [None, None]
            mdl["holdout_gap"] = None
            mdl["holdout_score"] = None
            mdl["reliability_pass_hat_k"] = {}
            mdl["premature_end_rate"] = None
            mdl["rank_band"] = None
            # Macro + failure profile absent on legacy parquets (#55).
            mdl["efficacy_macro_category"] = None
            mdl["efficacy_macro_category_ci"] = [None, None]
            mdl["failure_profile"] = None
            # Consistency band is OMITTED (not nulled) on single-run parquets (#71).
            mdl.pop("consistency_band", None)
        lb["statistical_note"] = None
        # Corpus health is omitted entirely when uncomputable (#71).
        lb.pop("corpus_health", None)
    return lb


@_skip_no_node
class TestHeadlessRender:
    def test_empty_state_renders_intentional_panel(self):
        out = _run_node("empty", None)
        assert out["ok"], out.get("error")
        assert out["empty_shown"] is True
        assert out["results_hidden"] is True

    def test_populated_state_renders_rows_and_scatter(self):
        out = _run_node("populated", _populated_payload(degraded=False))
        assert out["ok"], out.get("error")
        # One <tr> per model in the table body.
        assert out["body_html"].count("<tr") == 3
        assert out["cards_html"].count("card-head") == 3
        # Scatter drew axes + dots (well more than just the kept <title>).
        assert out["svg_children"] > 3
        # Holdout column visible (this payload has a holdout).
        assert out["holdout_th_display"] == ""
        # Value column computed a finite number for every model.
        assert all(v is not None for v in out["values"])
        # Pareto frontier is non-empty and a subset of the model names.
        assert out["frontier"], "expected a non-empty Pareto frontier"

    def test_degraded_state_renders_without_throwing(self):
        out = _run_node("populated", _populated_payload(degraded=True))
        assert out["ok"], out.get("error")
        # Still one row per model even with null CIs / holdout / pass^k.
        assert out["body_html"].count("<tr") == 3
        # No holdout column (degraded payload drops the holdout half).
        assert out["holdout_th_display"] == "none"
        # CI whisker / pass^k detail simply omitted — no "NaN"/"undefined" leak.
        assert "undefined" not in out["body_html"]
        assert "NaN" not in out["body_html"]

    def test_diagnostics_render_on_full_payload(self):
        # Full payload (failure profiles + stratified sim profiles + corpus
        # health): all three diagnostic sections render, panel visible.
        out = _run_node("populated", _populated_payload(degraded=False))
        assert out["ok"], out.get("error")
        assert out["diag_panel_display"] == ""
        assert out["diag_failure_display"] == ""
        assert out["diag_robustness_display"] == ""
        assert out["diag_corpus_display"] == ""
        # Failure table: a taxonomy mode and a percentage rendered.
        assert "incomplete" in out["diag_html"]
        assert "%" in out["diag_html"]
        # Robustness table: non-cooperative profile columns + delta vs coop.
        assert "impatient" in out["diag_html"]
        assert "vs coop" in out["diag_html"]
        # Corpus health: count tiles with the defect/low-signal explanations.
        assert "never passed" in out["diag_html"]
        assert "passed by every model" in out["diag_html"]
        # Nothing un-numeric ever leaks into the diagnostics markup.
        assert "undefined" not in out["diag_html"]
        assert "NaN" not in out["diag_html"]

    def test_diagnostics_hidden_on_degraded_payload(self):
        # Legacy/degraded payload: no failure profiles, no sim_profile_robustness,
        # no corpus_health — every section and the whole panel stay hidden, and
        # nothing renders into the diagnostic sinks.
        out = _run_node("populated", _populated_payload(degraded=True))
        assert out["ok"], out.get("error")
        assert out["diag_panel_display"] == "none"
        assert out["diag_failure_display"] == "none"
        assert out["diag_robustness_display"] == "none"
        assert out["diag_corpus_display"] == "none"
        assert "undefined" not in out["diag_html"]
        assert "NaN" not in out["diag_html"]

    def test_demo_payload_renders_diagnostics(self):
        # ?demo=1 fabricated payload must exercise every diagnostics section so
        # the panel can be worked on without real results.
        out = _run_node("demo", None)
        assert out["ok"], out.get("error")
        assert out["body_html"].count("<tr") == 6
        assert out["diag_panel_display"] == ""
        assert out["diag_failure_display"] == ""
        assert out["diag_robustness_display"] == ""
        assert out["diag_corpus_display"] == ""
        assert "adversarial" in out["diag_html"]
        assert "undefined" not in out["diag_html"]
        assert "NaN" not in out["diag_html"]

    def test_diagnostics_escape_untrusted_strings(self):
        # Model names, failure-mode names (taxonomy can grow out-of-vocabulary),
        # profile names, and the corpus-health headline all flow into innerHTML
        # sinks in the diagnostics panel; every one must pass through esc().
        # The payload carries a double quote so ATTRIBUTE-context escaping
        # (title="..." tooltips) is pinned too, not just element context — a
        # quote-less payload would pass even if esc() stopped escaping quotes.
        evil = '<b "x>xss</b>'
        lb = _populated_payload(degraded=False)
        victim = lb["models"][0]["name"]
        lb["models"][0]["name"] = evil
        lb["models"][0]["failure_profile"]["modes"][evil] = {"count": 3, "rate": 0.05}
        spr = lb["sim_profile_robustness"]["models"]
        spr[evil] = spr.pop(victim)
        profiles = lb["sim_profile_robustness"]["models"][evil]
        profiles[evil] = dict(profiles["impatient"])
        lb["corpus_health"]["headline"] = evil
        out = _run_node("populated", lb)
        assert out["ok"], out.get("error")
        assert "<b" not in out["diag_html"]
        assert '"x' not in out["diag_html"].replace("&quot;x", "")
        assert "&lt;b &quot;x&gt;xss&lt;/b&gt;" in out["diag_html"]
