/* Fills the datasheet from assets/report.json (written by adsrank/export_site.py). */
(function () {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const C = window.Charts;
  const f = {
    int: (v) => Math.round(v).toLocaleString("en-US"),
    pct: (v, d = 1) => (v >= 0.99995 ? "100" : (100 * v).toFixed(d)) + "%",
    ms: (v, d = 1) => v.toFixed(d) + " ms",
    d: (v, n = 4) => Number(v).toFixed(n),
    bytes: (b) => (b > 1e6 ? (b / 1e6).toFixed(1) + " MB" : (b / 1e3).toFixed(0) + " kB"),
  };
  const ARCH_NAME = { lr: "Logistic regression", fm: "FM", dnn: "DNN (linear + MLP)", deepfm: "DeepFM" };

  // ---- theme toggle ----
  const root = document.documentElement;
  function setTheme(t) {
    if (t) root.setAttribute("data-theme", t); else root.removeAttribute("data-theme");
    $$("[data-theme-set]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.themeSet === t)));
    try { t ? localStorage.setItem("ark-theme", t) : localStorage.removeItem("ark-theme"); } catch (e) { /* storage blocked */ }
  }
  try { const saved = localStorage.getItem("ark-theme"); if (saved) setTheme(saved); } catch (e) { /* ignore */ }
  $$("[data-theme-set]").forEach((b) => b.addEventListener("click", () =>
    setTheme(root.getAttribute("data-theme") === b.dataset.themeSet ? null : b.dataset.themeSet)));

  fetch("assets/report.json").then((r) => r.json()).then(render).catch((e) => {
    console.error(e);
    $$(".loading").forEach((n) => (n.textContent = "report.json missing — run python -m adsrank.export_site"));
  });

  function bestTuned(cpp) {
    return cpp.filter((r) => r.mode === "tuned").reduce((a, b) => (b.p95 < a.p95 ? b : a));
  }

  function render(R) {
    window.REPORT = R;
    const runs = R.ranker.runs;
    const ret = R.retrieval;
    const head = ret.headline.two_tower_distilled;
    const headClicks = ret.headline.two_tower_clicks;
    const best = bestTuned(R.latency.cpp);
    const naiveDefault = R.latency.cpp.find((r) => r.mode === "naive" && r.threads === 0);
    const eager = R.latency.py.pytorch_eager;

    const K = {
      month: R.month, generated: R.generated, rows: f.int(R.dataset.rows),
      auc_lr: f.d(runs.lr.test.auc), auc_deepfm: f.d(runs.deepfm.test.auc),
      ll_rel: f.pct((runs.lr.test.logloss - runs.deepfm.test.logloss) / runs.lr.test.logloss),
      ret_head: f.pct(head.retention), ret_fewer: f.pct(1 - ret.headline_frac, 0),
      ...(() => { const t = ret.curves.two_tower_distilled.find((c) => Math.abs(c.frac - 0.002) < 1e-9) || ret.curves.two_tower_distilled[0];
        const p = ret.curves.ranker_prior.find((c) => c.k === t.k);
        return { tight_frac: f.pct(t.frac, 1), tight_k: t.k, tight_ret: f.pct(t.retention), tight_prior: f.pct(p.retention) }; })(),
      lat_base: f.ms(eager.p95), lat_best: f.ms(best.p95), batch: f.int(R.export.batch),
      min_count: R.dataset.min_count, n_test: f.int(R.dataset.split_sizes.test),
      k_head: f.int(head.k), n_items: f.int(ret.n_items), emb_rows: f.int(R.dataset.emb_rows),
      item_key: R.field_split.key, u_thresh: R.field_split.threshold,
      item_fields: R.field_split.item_fields.join(", "),
      faiss_dim: ret.faiss.dim, faiss_nlist: ret.faiss.nlist, faiss_nprobe: ret.faiss.nprobe,
      faiss_recall: f.pct(head.faiss_recall_vs_exact, 1),
      parity: R.export.parity.max_abs_diff.toExponential(1), n_req: R.export.n_requests, ort_version: R.export.ort_version,
    };
    $$("[data-k]").forEach((n) => { if (K[n.dataset.k] !== undefined) n.textContent = K[n.dataset.k]; });

    // ---- device info ----
    $("#devinfo").innerHTML = R.artifacts.filter((a) => a.devinfo).map((a) =>
      `<tr><td><b>${a.devinfo}</b></td><td><code>${a.name}</code></td><td class="r">${f.bytes(a.bytes)}</td></tr>`).join("");

    schematic(R, head);
    toc();
    $("#revhist").innerHTML = R.revisions.map((r) => `<tr><td><b>${r.change}</b></td><td>${r.desc}</td></tr>`).join("");
    pinout(R);

    // ---- 6.1 operating conditions ----
    $("#roc tbody").innerHTML = R.operating.map((r) =>
      `<tr><td>${r[0]}</td><td>${r[1]}</td><td class="c">${r[2]}</td></tr>`).join("");

    // ---- 6.2 characteristics ----
    const seeds = R.ranker.seeds;
    const mm = (arr, k) => arr && arr.length > 1 ? [Math.min(...arr.map((s) => s[k])), Math.max(...arr.map((s) => s[k]))] : [null, null];
    const [aMin, aMax] = mm(seeds.deepfm, "auc"), [lMin, lMax] = mm(seeds.deepfm, "logloss");
    const [bMin, bMax] = mm(seeds.lr, "auc");
    const nSeeds = (seeds.deepfm || []).length;
    const dash = "—";
    const rows = [
      ["grp", "Ranker (§7.3.2)"],
      ["Test AUC, DeepFM", `${K.n_test} held-out impressions; ${nSeeds} seeds<sup class="fn">(1)</sup>`, aMin, runs.deepfm.test.auc, aMax, "", 4],
      ["Test log loss, DeepFM", "same", lMin, runs.deepfm.test.logloss, lMax, "", 4],
      ["Test AUC, logistic regression", `same features; ${(seeds.lr || []).length} seeds`, bMin, runs.lr.test.auc, bMax, "", 4],
      ["Log-loss reduction vs LR", "shipped checkpoints", null, K.ll_rel, null, ""],
      ["grp", "Retrieval funnel (§7.3.4–7.3.5)"],
      ["Retention@10, distilled two-tower", `K = ${K.k_head} of ${K.n_items}; ${f.int(ret.n_eval_queries)} queries<sup class="fn">(2)</sup>`, f.pct(head.retention_p10, 0), f.pct(head.retention), null, ""],
      ["Retention@10, click-trained two-tower", "same", f.pct(headClicks.retention_p10, 0), f.pct(headClicks.retention), null, ""],
      ["Retention@10, ranker-prior shortlist", "same; non-personalised", null, f.pct(ret.curves.ranker_prior.find((c) => c.k === head.k).retention), null, ""],
      ["Retention@10, popularity shortlist", "same", null, f.pct(ret.curves.popularity.find((c) => c.k === head.k).retention), null, ""],
      ["Ranker evaluations per request", "funnel vs. exhaustive", null, `${K.k_head} / ${K.n_items}`, null, "items"],
      ["FAISS IVF recall vs. exact search", `nlist ${K.faiss_nlist}, nprobe ${K.faiss_nprobe}`, null, K.faiss_recall, null, ""],
      ["FAISS search time", "per query, single thread batch", null, head.faiss_ms_per_query.toFixed(2), null, "ms"],
      ["grp", "Serving (§7.3.6), batch " + K.batch],
      ["p95 latency, PyTorch eager", `${R.latency.py.torch_threads} torch threads`, null, eager.p95.toFixed(2), null, "ms"],
      ["p95 latency, ORT C++ naive", "ORT default threads", null, naiveDefault.p95.toFixed(2), null, "ms"],
      ["p95 latency, ORT C++ tuned", `${best.threads} intra-op threads, IoBinding`, null, best.p95.toFixed(2), null, "ms"],
      ["p50 / p99 latency, ORT C++ tuned", "same", best.p50.toFixed(2), null, best.p99.toFixed(2), "ms"],
      ["ONNX vs PyTorch max |Δp|", "20,000 held-out rows", null, K.parity, null, ""],
    ];
    const cell = (v, d) => v === null || v === undefined ? dash : typeof v === "number" ? v.toFixed(d ?? 4) : v;
    $("#chars tbody").innerHTML = rows.map((r) => r[0] === "grp"
      ? `<tr class="grp"><td colspan="6">${r[1]}</td></tr>`
      : `<tr><td>${r[0]}</td><td>${r[1]}</td><td class="c">${cell(r[2], r[6])}</td><td class="c"><b>${cell(r[3], r[6])}</b></td><td class="c">${cell(r[4], r[6])}</td><td class="c">${r[5]}</td></tr>`).join("") +
      `<tr><td colspan="6" class="tnote">(1) TYP is the shipped seed-1 checkpoint (the one exported and benchmarked); MIN/MAX span all seeds. (2) MIN column for retention is the 10th-percentile query.</td></tr>`;

    // ---- 6.3 ablation ----
    const yes = "●", no = "○";
    $("#ablation tbody").innerHTML = ["lr", "fm", "dnn", "deepfm"].map((a) => {
      const r = runs[a];
      const d = (r.test.logloss - runs.lr.test.logloss) / runs.lr.test.logloss;
      return `<tr${a === "deepfm" ? ' class="is-hot"' : ""}><td><b>${ARCH_NAME[a]}</b></td><td class="c">${yes}</td><td class="c">${a === "fm" || a === "deepfm" ? yes : no}</td><td class="c">${a === "dnn" || a === "deepfm" ? yes : no}</td><td class="r">${f.int(r.params)}</td><td class="c">${r.l2 ? r.l2.toExponential(0) : "0"}</td><td class="r">${r.test.auc.toFixed(4)}</td><td class="r">${r.test.logloss.toFixed(5)}</td><td class="r">${a === "lr" ? "—" : (d * 100).toFixed(2) + "%"}</td></tr>`;
    }).join("");

    {
      const r = runs, gain = (a, b) => r[a].test.auc - r[b].test.auc;
      const llg = (a, b) => 100 * (r[b].test.logloss - r[a].test.logloss) / r[b].test.logloss;
      const sgn = (v, d = 4) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d);
      $("#ablation-note").innerHTML = `<b>Readout</b>Second-order FM terms alone lift AUC ${sgn(gain("fm", "lr"))} over LR; the deep MLP alone lifts it ${sgn(gain("dnn", "lr"))}. ` +
        `Adding the FM term on top of the MLP (DeepFM vs DNN) ${Math.abs(gain("deepfm", "dnn")) < 5e-5 ? "leaves AUC unchanged" : "moves AUC " + sgn(gain("deepfm", "dnn"))} and moves log loss ${sgn(-llg("deepfm", "dnn"), 2)}% — on this sample almost all of DeepFM's gain comes from the deep tower, and the FM term mainly lets it reach that point in fewer epochs (best epoch ${r.deepfm.best_epoch} vs ${r.dnn.best_epoch}).`;
    }
    // entity probe table + ranker-prior caveat
    if (R.probe) {
      $("#probe tbody").innerHTML = R.probe.candidates.map((c) => `<tr${c.key === R.probe.selected ? ' class="is-hot"' : ""}><td><b>${c.key}</b>${c.key === R.probe.selected ? " ◂ selected" : ""}</td><td>${c.item_fields.join(", ")}</td><td class="r">${f.pct(c.distinct_top10, 1)}</td><td class="r">${f.pct(c.prior_ret_1pct, 1)}</td></tr>`).join("");
    }
    {
      const cv = ret.curves, at = (k, fr) => cv[k].find((c) => Math.abs(c.frac - fr) < 1e-9);
      const lo = cv.two_tower_distilled[0].frac;  // tightest budget: where the shortlist actually has to be right
      const pr = at("ranker_prior", lo), di = at("two_tower_distilled", lo), cl = at("two_tower_clicks", ret.headline_frac);
      $("#prior-note").innerHTML = `<b>Read this first</b>On Criteo, DeepFM orders catalog items largely by item main effects: one global shortlist of the top ${f.pct(lo, 1)} of items (by mean DeepFM logit) already retains ${f.pct(pr.retention)} of each query's top-10, and ${f.pct(at("ranker_prior", ret.headline_frac).retention)} at the ${f.pct(ret.headline_frac, 0)} headline budget. ` +
        `The distilled tower's job is the query-dependent remainder: at ${f.pct(lo, 1)} it retains ${f.pct(di.retention)}, cutting the global list's misses by ${f.pct(1 - (1 - di.retention) / Math.max(1e-9, 1 - pr.retention), 0)}. The click-trained tower — the textbook retrieval objective — retains only ${f.pct(cl.retention)} at the headline budget: it learns which ads were <i>served</i> to which contexts, not which ones the ranker scores highest.`;
    }
    figures(R, best);
    blockdiag(R, head);
    pkg(R);
    document.dispatchEvent(new CustomEvent("report-ready", { detail: R }));
  }

  function toc() {
    $("#toc").innerHTML = $$("h2[id], h3[id]").map((h) => {
      const lvl = h.tagName === "H3" ? "l2" : "";
      const label = h.textContent.replace(/^\s*([\d.]+)\s*/, "$1 ");
      return `<a class="${lvl}" href="#${h.id}"><span>${label}</span><span class="dots"></span></a>`;
    }).join("");
  }

  // ---- simplified schematic (page 1) ----
  function schematic(R, head) {
    const s = $("#schematic");
    const n = R.retrieval.n_items, k = head.k;
    s.innerHTML = `
      <defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto-start-reverse"><path d="M0,0L10,5L0,10z" class="arrow"/></marker></defs>
      <text x="8" y="54" class="sm">REQUEST</text>
      <text x="8" y="70" class="lbl">context</text><text x="8" y="86" class="sm">34 pins</text>
      <path d="M70,76 H128" class="bus" marker-end="url(#ah)"/>
      <rect x="132" y="40" width="130" height="72" class="blk"/><text x="197" y="70" text-anchor="middle" class="lbl">QUERY TOWER</text><text x="197" y="88" text-anchor="middle" class="sm">MLP → ${R.retrieval.faiss.dim - 1}-d</text>
      <path d="M262,76 H318" class="wire" marker-end="url(#ah)"/>
      <rect x="322" y="40" width="150" height="72" class="blk"/><text x="397" y="70" text-anchor="middle" class="lbl">FAISS IVF</text><text x="397" y="88" text-anchor="middle" class="sm">inner product, top-${k}</text>
      <rect x="322" y="140" width="150" height="40" class="blk"/><text x="397" y="164" text-anchor="middle" class="sm">${n.toLocaleString()} item vectors</text>
      <path d="M397,140 V116" class="wire" marker-end="url(#ah)"/>
      <path d="M472,76 H540" class="bus" marker-end="url(#ah)"/><text x="506" y="66" text-anchor="middle" class="sm">${k}</text>
      <rect x="544" y="34" width="170" height="84" class="blk hot"/><text x="629" y="68" text-anchor="middle" class="lbl">DeepFM RANKER</text><text x="629" y="86" text-anchor="middle" class="sm">ONNX Runtime · C++</text><text x="629" y="102" text-anchor="middle" class="sm">${k} pCTR / request</text>
      <path d="M714,76 H790" class="bus" marker-end="url(#ah)"/>
      <rect x="794" y="52" width="86" height="48" class="blk"/><text x="837" y="81" text-anchor="middle" class="lbl">TOP-10</text>
      <text x="890" y="70" class="sm">retains</text><text x="890" y="88" class="lbl acc">${(100 * head.retention).toFixed(0)}%</text><text x="890" y="104" class="sm">of oracle</text>
      <path d="M629,118 V160 H472" class="wire" stroke-dasharray="4 3"/>
      <text x="640" y="152" class="sm">distillation target (training only)</text>`;
  }

  function blockdiag(R, head) {
    const s = $("#blockdiag");
    const it = R.field_split.item_fields, nI = it.length;
    s.innerHTML = `
      <defs><marker id="ah2" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto-start-reverse"><path d="M0,0L10,5L0,10z" class="arrow"/></marker></defs>
      <rect x="6" y="20" width="948" height="130" fill="none" stroke="var(--ink-3)" stroke-dasharray="5 4"/><text x="14" y="36" class="sm">ONLINE PATH (per request)</text>
      <rect x="6" y="170" width="948" height="122" fill="none" stroke="var(--ink-3)" stroke-dasharray="5 4"/><text x="14" y="186" class="sm">OFFLINE (training / index build)</text>
      <text x="20" y="80" class="lbl">I1–I13</text><text x="20" y="96" class="sm">+ ${34 - 13} cat. pins</text>
      <path d="M86,84 H120" class="bus" marker-end="url(#ah2)"/>
      <rect x="124" y="54" width="112" height="60" class="blk"/><text x="180" y="80" text-anchor="middle" class="lbl">EMBED 16-d</text><text x="180" y="96" text-anchor="middle" class="sm">34 fields</text>
      <path d="M236,84 H262" class="wire" marker-end="url(#ah2)"/>
      <rect x="266" y="54" width="112" height="60" class="blk"/><text x="322" y="80" text-anchor="middle" class="lbl">MLP 256·128</text><text x="322" y="96" text-anchor="middle" class="sm">→ unit 64-d ⊕ 1</text>
      <path d="M378,84 H404" class="wire" marker-end="url(#ah2)"/>
      <rect x="408" y="54" width="120" height="60" class="blk"/><text x="468" y="80" text-anchor="middle" class="lbl">IVF SEARCH</text><text x="468" y="96" text-anchor="middle" class="sm">nprobe ${R.retrieval.faiss.nprobe}, K=${head.k}</text>
      <path d="M528,84 H554" class="bus" marker-end="url(#ah2)"/>
      <rect x="558" y="48" width="150" height="72" class="blk hot"/><text x="633" y="72" text-anchor="middle" class="lbl">DeepFM</text><text x="633" y="88" text-anchor="middle" class="sm">linear + FM + MLP 400³</text><text x="633" y="104" text-anchor="middle" class="sm">int64[K,39] → p[K]</text>
      <path d="M708,84 H734" class="wire" marker-end="url(#ah2)"/>
      <rect x="738" y="54" width="90" height="60" class="blk"/><text x="783" y="80" text-anchor="middle" class="lbl">TOP-N</text><text x="783" y="96" text-anchor="middle" class="sm">partial sort</text>
      <path d="M828,84 H866" class="wire" marker-end="url(#ah2)"/><text x="872" y="88" class="lbl">10 ads</text>
      <rect x="124" y="206" width="160" height="60" class="blk"/><text x="204" y="232" text-anchor="middle" class="lbl">ITEM TOWER</text><text x="204" y="248" text-anchor="middle" class="sm">${nI} ad pins (${it.join(", ")})</text>
      <path d="M284,236 H330" class="wire" marker-end="url(#ah2)"/>
      <rect x="334" y="206" width="150" height="60" class="blk"/><text x="409" y="232" text-anchor="middle" class="lbl">ITEM VECTORS</text><text x="409" y="248" text-anchor="middle" class="sm">${R.retrieval.n_items.toLocaleString()} × ${R.retrieval.faiss.dim} (+bias)</text>
      <path d="M468,206 V118" class="wire" marker-end="url(#ah2)"/>
      <rect x="558" y="206" width="150" height="60" class="blk"/><text x="633" y="232" text-anchor="middle" class="lbl">LISTWISE KL</text><text x="633" y="248" text-anchor="middle" class="sm">teacher = DeepFM logits</text>
      <path d="M633,206 V124" class="wire" stroke-dasharray="4 3"/>
      <path d="M558,236 H488" class="wire" stroke-dasharray="4 3" marker-end="url(#ah2)"/>
      <rect x="738" y="206" width="200" height="60" class="blk"/><text x="838" y="232" text-anchor="middle" class="lbl">ONNX EXPORT</text><text x="838" y="248" text-anchor="middle" class="sm">opset 18 · parity ${R.export.parity.max_abs_diff.toExponential(0)}</text>
      <path d="M738,236 H720 V124" class="wire" marker-end="url(#ah2)"/>`;
  }

  // ---- pinout ----
  function pinout(R) {
    const s = $("#pinout");
    const fields = R.fields;
    const pins = fields.map((fd, i) => ({ ...fd, no: i + 1 }));
    pins.push({ name: "pCTR", kind: "out", side: "output", no: 40, card: null, desc: R.pin40.desc });
    const top = 50, pitch = 26, bodyX = 140, bodyW = 160;
    let h = `<rect x="${bodyX}" y="${top - 22}" width="${bodyW}" height="${20 * pitch + 20}" rx="4" class="pkg"/>
      <path d="M${bodyX + bodyW / 2 - 16},${top - 22} a16,16 0 0 0 32,0" fill="var(--paper)" stroke="var(--ink)" stroke-width="1.5"/>
      <circle cx="${bodyX + 16}" cy="${top}" r="4" fill="var(--ink)"/>
      <text x="${bodyX + bodyW / 2}" y="${top + 9 * pitch}" text-anchor="middle" style="font:800 20px var(--sans)">ARK-3</text>
      <text x="${bodyX + bodyW / 2}" y="${top + 9 * pitch + 18}" text-anchor="middle" class="pnum" style="font-size:11px">DeepFM · 39 IN</text>
      <text x="${bodyX + bodyW / 2}" y="${top + 9 * pitch + 32}" text-anchor="middle" class="pnum" style="font-size:11px">${R.dataset.emb_rows.toLocaleString()} EMB ROWS</text>
      <text x="${bodyX + bodyW / 2}" y="${top + 11 * pitch + 10}" text-anchor="middle" class="pnum" style="font-size:10px;letter-spacing:.1em">TOP VIEW</text>`;
    pins.forEach((p) => {
      const left = p.no <= 20;
      const row = left ? p.no - 1 : 40 - p.no;
      const y = top + row * pitch;
      const cls = p.kind === "out" ? "pin out" : p.side === "item" ? "pin item" : "pin";
      const px = left ? bodyX - 34 : bodyX + bodyW;
      h += `<g class="p" data-no="${p.no}">
        <rect x="${px}" y="${y - 6}" width="34" height="12" class="${cls}"/>
        <text x="${left ? bodyX + 8 : bodyX + bodyW - 8}" y="${y + 4}" class="pnum" text-anchor="${left ? "start" : "end"}">${p.no}</text>
        <text x="${left ? px - 6 : px + 40}" y="${y + 4}" class="pname" text-anchor="${left ? "end" : "start"}" ${p.side === "item" ? 'style="fill:var(--accent)"' : ""}>${p.name}</text>
      </g>`;
    });
    s.innerHTML = h;
    const tb = $("#pintable tbody");
    tb.innerHTML = pins.map((p) => `<tr data-no="${p.no}"><td class="c">${p.no}</td><td><b ${p.side === "item" ? 'class="hl"' : ""}>${p.name}</b></td><td class="c">${p.kind === "out" ? "O" : "I"}</td><td>${p.kind === "out" ? "—" : p.side === "item" ? "Ad" + (p.name === R.field_split.key ? " (key)" : "") : "Context"}</td><td class="r">${p.card ? p.card.toLocaleString() : "—"}</td><td>${p.desc}</td></tr>`).join("");
    const hot = (no, on) => {
      $$(`[data-no="${no}"]`).forEach((n) => n.classList.toggle("is-hot", on));
    };
    $$("g.p", s).forEach((g) => {
      const no = +g.dataset.no, p = pins[no - 1];
      g.addEventListener("mouseenter", (ev) => {
        hot(no, true);
        const row = $(`#pintable tr[data-no="${no}"]`);
        const wrap = row.closest(".tbl-wrap");
        wrap.scrollTop = row.offsetTop - wrap.clientHeight / 2;
      });
      g.addEventListener("mousemove", (ev) => C.showTip(`<b>Pin ${no} · ${p.name}</b><div>${p.desc}</div>${p.card ? `<div class="row"><span>embedding rows</span><b>${p.card.toLocaleString()}</b></div>` : ""}${p.oov !== undefined && p.oov !== null ? `<div class="row"><span>OOV rate</span><b>${(100 * p.oov).toFixed(1)}%</b></div>` : ""}`, ev));
      g.addEventListener("mouseleave", () => { hot(no, false); C.hideTip(); });
    });
    $$("tr[data-no]", tb).forEach((tr) => {
      tr.addEventListener("mouseenter", () => hot(+tr.dataset.no, true));
      tr.addEventListener("mouseleave", () => hot(+tr.dataset.no, false));
    });
  }

  // ---- figures ----
  function figures(R, best) {
    const box = $("#figs");
    box.innerHTML = "";
    const runs = R.ranker.runs;
    const order = ["deepfm", "dnn", "fm", "lr"];

    // 6-1 validation AUC vs epoch
    const allAuc = order.flatMap((a) => runs[a].curve.map((c) => c.auc));
    const maxEp = Math.max(...order.flatMap((a) => runs[a].curve.map((c) => c.epoch)));
    C.line(box, {
      id: "fig-6-1", caption: "Figure 6-1. Validation AUC vs Training Epoch",
      cond: "Evaluated 4× per epoch; the best-validation checkpoint is shipped",
      x: { label: "Epoch", min: 0, max: Math.ceil(maxEp * 2) / 2, fmt: (v) => String(+v.toFixed(2)), tipFmt: (v) => v.toFixed(2) },
      y: { label: "Validation AUC", min: Math.floor(Math.min(...allAuc) * 200) / 200, max: Math.ceil(Math.max(...allAuc) * 200) / 200, fmt: (v) => v.toFixed(3), tipFmt: (v) => v.toFixed(4) },
      series: order.map((a) => ({ name: ARCH_NAME[a].replace(" (linear + MLP)", ""), pts: runs[a].curve.map((c) => [c.epoch, c.auc]) })),
      legendAt: "br",
      table: { head: ["Epoch", ...order.map((a) => a.toUpperCase())], rows: [...new Set(order.flatMap((a) => runs[a].curve.map((c) => c.epoch)))].sort((a, b) => a - b).map((e) => [e.toFixed(2), ...order.map((a) => { const c = runs[a].curve.find((q) => q.epoch === e); return c ? c.auc.toFixed(4) : "—"; })]) },
    });

    // 6-2 calibration
    const cal = runs.deepfm.calibration;
    const lo = Math.min(...cal.flatMap((c) => [c.pred, c.obs])), hi = Math.max(...cal.flatMap((c) => [c.pred, c.obs]));
    const lmin = Math.pow(10, Math.floor(Math.log10(lo) * 2) / 2), lmax = Math.pow(10, Math.ceil(Math.log10(hi) * 2) / 2);
    const lt = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5].filter((v) => v >= lmin && v <= lmax);
    C.line(box, {
      id: "fig-6-2", caption: "Figure 6-2. DeepFM Reliability (Calibration)",
      cond: `${cal.length} equal-mass bins over ${R.dataset.split_sizes.test.toLocaleString()} test impressions`,
      x: { label: "Mean predicted CTR (log)", min: lmin, max: lmax, log: true, ticks: lt, fmt: (v) => (100 * v).toFixed(v < 0.01 ? 1 : 0) + "%", tipFmt: (v) => (100 * v).toFixed(2) + "%" },
      y: { label: "Observed CTR (log)", min: lmin, max: lmax, log: true, ticks: lt, fmt: (v) => (100 * v).toFixed(v < 0.01 ? 1 : 0) + "%", tipFmt: (v) => (100 * v).toFixed(2) + "%" },
      diagonal: true,
      series: [{ name: "DeepFM", pts: cal.map((c) => [c.pred, c.obs]) }],
      legendAt: false, condLines: ["diagonal = perfect calibration"], condAt: "tl",
      table: { head: ["Predicted", "Observed"], rows: cal.map((c) => [(100 * c.pred).toFixed(2) + "%", (100 * c.obs).toFixed(2) + "%"]) },
    });

    // 6-3 retention vs budget
    const ret = R.retrieval;
    const rs = [["two_tower_distilled", "Two-tower, ranker-distilled"], ["ranker_prior", "Ranker prior (same list for all)"], ["two_tower_clicks", "Two-tower, click-trained"], ["popularity", "Popularity (train frequency)"]];
    C.line(box, {
      id: "fig-6-3", caption: "Figure 6-3. Retention@10 vs Candidate Budget",
      cond: `${ret.n_eval_queries.toLocaleString()} held-out queries · ${ret.n_items.toLocaleString()}-item catalog · FAISS IVF`,
      x: { label: "Candidates scored by ranker (% of catalog, log scale)", log: true, min: 0.08, max: Math.max(...ret.curves.popularity.map((c) => c.frac)) * 100 * 1.15,
           ticks: ret.curves.popularity.map((c) => +(100 * c.frac).toFixed(1)), fmt: (v) => v + "%", tipFmt: (v) => v + "%" },
      y: { label: "Retention@10", min: 0, max: 1, fmt: (v) => (100 * v).toFixed(0) + "%", tipFmt: (v) => (100 * v).toFixed(1) + "%" },
      series: rs.map(([k, n]) => ({ name: n, pts: ret.curves[k].map((c) => [+(100 * c.frac).toFixed(1), c.retention]) })),
      refs: [{ x: ret.headline_frac * 100, label: `K = ${ret.headline.two_tower_distilled.k}` }],
      legendAt: { x: 0.03, y: 0.56 },
      table: { head: ["Budget", "K", ...rs.map((r) => r[1])], rows: ret.curves.popularity.map((c, i) => [(100 * c.frac).toFixed(1) + "%", c.k, ...rs.map(([k]) => (100 * ret.curves[k][i].retention).toFixed(1) + "%")]) },
    });

    // 6-4 latency vs threads
    const cpp = R.latency.cpp.filter((r) => r.threads > 0);
    const th = [...new Set(cpp.map((r) => r.threads))].sort((a, b) => a - b);
    const tuned = th.map((t) => [t, cpp.find((r) => r.mode === "tuned" && r.threads === t).p95]);
    const naive = th.map((t) => [t, cpp.find((r) => r.mode === "naive" && r.threads === t).p95]);
    const eager = R.latency.py.pytorch_eager.p95;
    const ymax = Math.max(eager, ...tuned.map((p) => p[1]), ...naive.map((p) => p[1])) * 1.12;
    C.line(box, {
      id: "fig-6-4", caption: "Figure 6-4. p95 Ranking Latency vs Intra-Op Threads",
      cond: `ONNX Runtime ${R.export.ort_version} C++ · batch ${R.export.batch} · ${R.latency.cpp[0].iters.toLocaleString()} requests per point`,
      x: { label: "Intra-op threads", min: 0, max: Math.max(...th) + 1, ticks: th, tipLabel: "Threads" },
      y: { label: "p95 latency (ms)", min: 0, max: ymax, fmt: (v) => v.toFixed(0), tipFmt: (v) => v.toFixed(2) + " ms" },
      series: [{ name: "C++ tuned (IoBinding)", pts: tuned }, { name: "C++ naive (alloc per call)", pts: naive }],
      refs: [{ y: eager, label: `PyTorch eager p95 ${eager.toFixed(1)} ms`, right: true }],
      legendAt: "tr", margin: { t: 14 },
      table: { head: ["Threads", "Tuned p50", "Tuned p95", "Naive p50", "Naive p95"], rows: th.map((t) => { const a = cpp.find((r) => r.mode === "tuned" && r.threads === t), b = cpp.find((r) => r.mode === "naive" && r.threads === t); return [t, a.p50.toFixed(2), a.p95.toFixed(2), b.p50.toFixed(2), b.p95.toFixed(2)]; }) },
    });

    // 6-5 serving paths
    const nd = R.latency.cpp.find((r) => r.mode === "naive" && r.threads === 0);
    const paths = [
      { name: "PyTorch eager", ...R.latency.py.pytorch_eager, sub: "Python, nn.Module" },
      { name: "ORT · Python", ...R.latency.py.ort_python, sub: "InferenceSession defaults" },
      { name: "ORT · C++ naive", ...nd, sub: "default threads, alloc per call" },
      { name: `ORT · C++ tuned`, ...best, accent: true, sub: `${best.threads} threads, IoBinding` },
    ];
    C.ranges(box, {
      id: "fig-6-5", caption: "Figure 6-5. Latency Distribution by Serving Path",
      cond: `batch ${R.export.batch} (one request) · ${R.export.n_requests} distinct replayed requests`,
      x: { label: "Latency per request (ms)", min: 0, max: Math.ceil(Math.max(...paths.map((p) => p.p99)) * 1.15 * 2) / 2, fmt: (v) => String(+v.toFixed(2)), labelFmt: (v) => v.toFixed(2) + " ms" },
      rows: paths,
      table: { head: ["Path", "p50", "p95", "p99", "mean"], rows: paths.map((p) => [p.name, p.p50.toFixed(2), p.p95.toFixed(2), p.p99.toFixed(2), p.mean.toFixed(2)]) },
    });

    // 6-6 field dependence heat map
    const fs = R.field_split;
    C.heat(box, {
      id: "fig-6-6", caption: "Figure 6-6. Field Dependence, 1 − U(J | K)",
      cond: `stronger shading = row field determines column field; red = selected ad entity (${fs.key} key)`,
      labels: fs.cat_fields, matrix: fs.U, hot: fs.item_fields, threshold: fs.threshold,
      table: { head: ["Key K", "Determines (U < " + fs.threshold + ")"], rows: fs.cat_fields.map((k, i) => [k, fs.cat_fields.filter((_, j) => j !== i && fs.U[i][j] < fs.threshold).join(" ") || "—"]) },
    });

    // 6-7 embedding L2 sweep
    const sw = R.ranker.sweeps;
    const l2s = sw.deepfm.map((r) => r.l2);
    C.line(box, {
      id: "fig-6-7", caption: "Figure 6-7. Validation Log Loss vs Embedding L2",
      cond: "Seed 1; lowest point per configuration is shipped",
      x: { label: "Embedding L2 coefficient", min: -0.4, max: l2s.length - 0.6, ticks: l2s.map((_, i) => i), fmt: (i) => (l2s[i] === 0 ? "0" : l2s[i].toExponential(0)), tipLabel: "L2" },
      y: (() => { const v = order.flatMap((a) => sw[a].map((r) => r.val.logloss)); return { label: "Validation log loss", min: Math.floor(Math.min(...v) * 2000) / 2000, max: Math.ceil(Math.max(...v) * 2000) / 2000, fmt: (x) => x.toFixed(4), tipFmt: (x) => x.toFixed(5) }; })(),
      series: order.map((a) => ({ name: ARCH_NAME[a].replace(" (linear + MLP)", ""), pts: sw[a].map((r, i) => [i, r.val.logloss]) })),
      legendAt: "tr",
      table: { head: ["L2", ...order.map((a) => a.toUpperCase())], rows: l2s.map((l, i) => [l === 0 ? "0" : l.toExponential(0), ...order.map((a) => sw[a][i].val.logloss.toFixed(5))]) },
    });

    // 6-8 distillation KL
    const log = ret.train_logs.two_tower_distilled;
    C.line(box, {
      id: "fig-6-8", caption: "Figure 6-8. Distillation Loss During Two-Tower Training",
      cond: "KL(DeepFM ‖ tower) over 128 uniform + 64 hard items per query",
      x: { label: "Optimizer step", min: 0, max: ret.distill_steps, fmt: (v) => (v >= 1000 ? v / 1000 + "k" : v), tipFmt: String },
      y: { label: "KL divergence (nats)", min: 0, max: Math.max(...log.map((l) => l.kl)) * 1.1, tipFmt: (v) => v.toFixed(4) },
      series: [{ name: "Distilled tower", pts: log.map((l) => [l.step, l.kl]) }],
      refs: [{ x: 400, label: "hard negatives on" }],
      legendAt: false,
      table: { head: ["Step", "KL"], rows: log.map((l) => [l.step, l.kl.toFixed(4)]) },
    });
  }

  function pkg(R) {
    $("#pkg tbody").innerHTML = R.artifacts.map((a) =>
      `<tr><td><code>${a.name}</code></td><td>${a.format}</td><td class="r">${f.bytes(a.bytes)}</td><td>${a.contents}</td></tr>`).join("");
  }
})();
