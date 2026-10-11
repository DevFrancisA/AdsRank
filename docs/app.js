/* Datasheet chrome: theme toggle, TOC, schematics, pinout. Reads the field
 * layout from assets/demo.json (written by adsrank/export_site.py). */
(function () {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];

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

  // ---- tooltip ----
  const tip = $("#tip");
  function showTip(html, ev) {
    tip.innerHTML = html;
    tip.style.display = "block";
    const w = tip.offsetWidth, h = tip.offsetHeight;
    let x = ev.clientX + 14, y = ev.clientY + 14;
    if (x + w > innerWidth - 8) x = ev.clientX - w - 14;
    if (y + h > innerHeight - 8) y = ev.clientY - h - 14;
    tip.style.left = x + "px";
    tip.style.top = y + "px";
  }
  const hideTip = () => (tip.style.display = "none");

  toc();
  schematic();
  fetch("assets/demo.json").then((r) => r.json()).then((D) => {
    $$("[data-k]").forEach((n) => {
      if (n.dataset.k === "item_key") n.textContent = D.key_field;
      if (n.dataset.k === "item_fields") n.textContent = D.item_fields.join(", ");
    });
    blockdiag(D);
    pinout(D);
  }).catch((e) => {
    console.error(e);
    $$(".loading").forEach((n) => (n.textContent = "demo.json missing — run python -m adsrank.export_site"));
  });

  function toc() {
    $("#toc").innerHTML = $$("h2[id], h3[id]").map((h) => {
      const lvl = h.tagName === "H3" ? "l2" : "";
      const label = h.textContent.replace(/^\s*([\d.]+)\s*/, "$1 ");
      return `<a class="${lvl}" href="#${h.id}"><span>${label}</span><span class="dots"></span></a>`;
    }).join("");
  }

  function arrowDef(id) {
    return `<defs><marker id="${id}" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" orient="auto-start-reverse"><path d="M0,0L10,5L0,10z" class="arrow"/></marker></defs>`;
  }

  function schematic() {
    $("#schematic").innerHTML = arrowDef("ah") + `
      <text x="8" y="54" class="sm">REQUEST</text>
      <text x="8" y="70" class="lbl">context</text><text x="8" y="86" class="sm">pins</text>
      <path d="M70,76 H128" class="bus" marker-end="url(#ah)"/>
      <rect x="132" y="40" width="130" height="72" class="blk"/><text x="197" y="70" text-anchor="middle" class="lbl">QUERY TOWER</text><text x="197" y="88" text-anchor="middle" class="sm">MLP → unit vector</text>
      <path d="M262,76 H318" class="wire" marker-end="url(#ah)"/>
      <rect x="322" y="40" width="150" height="72" class="blk"/><text x="397" y="70" text-anchor="middle" class="lbl">FAISS IVF</text><text x="397" y="88" text-anchor="middle" class="sm">inner product, top-K</text>
      <rect x="322" y="140" width="150" height="40" class="blk"/><text x="397" y="164" text-anchor="middle" class="sm">catalog item vectors</text>
      <path d="M397,140 V116" class="wire" marker-end="url(#ah)"/>
      <path d="M472,76 H540" class="bus" marker-end="url(#ah)"/><text x="506" y="66" text-anchor="middle" class="sm">K</text>
      <rect x="544" y="34" width="170" height="84" class="blk hot"/><text x="629" y="68" text-anchor="middle" class="lbl">DeepFM RANKER</text><text x="629" y="86" text-anchor="middle" class="sm">ONNX Runtime · C++</text><text x="629" y="102" text-anchor="middle" class="sm">pCTR per candidate</text>
      <path d="M714,76 H790" class="bus" marker-end="url(#ah)"/>
      <rect x="794" y="52" width="86" height="48" class="blk"/><text x="837" y="81" text-anchor="middle" class="lbl">TOP ADS</text>
      <path d="M629,118 V160 H472" class="wire" stroke-dasharray="4 3"/>
      <text x="640" y="152" class="sm">distillation target (training only)</text>`;
  }

  function blockdiag(D) {
    const it = D.item_fields;
    $("#blockdiag").innerHTML = arrowDef("ah2") + `
      <rect x="6" y="20" width="948" height="130" fill="none" stroke="var(--ink-3)" stroke-dasharray="5 4"/><text x="14" y="36" class="sm">ONLINE PATH (per request)</text>
      <rect x="6" y="170" width="948" height="122" fill="none" stroke="var(--ink-3)" stroke-dasharray="5 4"/><text x="14" y="186" class="sm">OFFLINE (training / index build)</text>
      <text x="20" y="80" class="lbl">CONTEXT</text><text x="20" y="96" class="sm">count + cat. pins</text>
      <path d="M86,84 H120" class="bus" marker-end="url(#ah2)"/>
      <rect x="124" y="54" width="112" height="60" class="blk"/><text x="180" y="80" text-anchor="middle" class="lbl">EMBED</text><text x="180" y="96" text-anchor="middle" class="sm">per-field tables</text>
      <path d="M236,84 H262" class="wire" marker-end="url(#ah2)"/>
      <rect x="266" y="54" width="112" height="60" class="blk"/><text x="322" y="80" text-anchor="middle" class="lbl">MLP</text><text x="322" y="96" text-anchor="middle" class="sm">→ unit vector ⊕ 1</text>
      <path d="M378,84 H404" class="wire" marker-end="url(#ah2)"/>
      <rect x="408" y="54" width="120" height="60" class="blk"/><text x="468" y="80" text-anchor="middle" class="lbl">IVF SEARCH</text><text x="468" y="96" text-anchor="middle" class="sm">shortlist of K</text>
      <path d="M528,84 H554" class="bus" marker-end="url(#ah2)"/>
      <rect x="558" y="48" width="150" height="72" class="blk hot"/><text x="633" y="72" text-anchor="middle" class="lbl">DeepFM</text><text x="633" y="88" text-anchor="middle" class="sm">linear + FM + MLP</text><text x="633" y="104" text-anchor="middle" class="sm">int64[K,39] → p[K]</text>
      <path d="M708,84 H734" class="wire" marker-end="url(#ah2)"/>
      <rect x="738" y="54" width="90" height="60" class="blk"/><text x="783" y="80" text-anchor="middle" class="lbl">TOP-N</text><text x="783" y="96" text-anchor="middle" class="sm">partial sort</text>
      <path d="M828,84 H866" class="wire" marker-end="url(#ah2)"/><text x="872" y="88" class="lbl">ads</text>
      <rect x="124" y="206" width="160" height="60" class="blk"/><text x="204" y="232" text-anchor="middle" class="lbl">ITEM TOWER</text><text x="204" y="248" text-anchor="middle" class="sm">ad pins (${it.join(", ")})</text>
      <path d="M284,236 H330" class="wire" marker-end="url(#ah2)"/>
      <rect x="334" y="206" width="150" height="60" class="blk"/><text x="409" y="232" text-anchor="middle" class="lbl">ITEM VECTORS</text><text x="409" y="248" text-anchor="middle" class="sm">vector ⊕ bias</text>
      <path d="M468,206 V118" class="wire" marker-end="url(#ah2)"/>
      <rect x="558" y="206" width="150" height="60" class="blk"/><text x="633" y="232" text-anchor="middle" class="lbl">LISTWISE KL</text><text x="633" y="248" text-anchor="middle" class="sm">teacher = DeepFM logits</text>
      <path d="M633,206 V124" class="wire" stroke-dasharray="4 3"/>
      <path d="M558,236 H488" class="wire" stroke-dasharray="4 3" marker-end="url(#ah2)"/>
      <rect x="738" y="206" width="200" height="60" class="blk"/><text x="838" y="232" text-anchor="middle" class="lbl">ONNX EXPORT</text><text x="838" y="248" text-anchor="middle" class="sm">parity-checked vs PyTorch</text>
      <path d="M738,236 H720 V124" class="wire" marker-end="url(#ah2)"/>`;
  }

  function pinout(D) {
    const item = new Set(D.item_idx);
    const pins = D.fields.map((name, i) => {
      const isItem = item.has(i), isKey = name === D.key_field;
      const desc = name.startsWith("I") ? "Count feature, log-squared bucketed"
        : isKey ? "Hashed categorical — ad key"
        : isItem ? `Hashed categorical — ad attribute (determined by ${D.key_field})`
        : "Hashed categorical (anonymised)";
      return { name, no: i + 1, side: isItem ? "item" : "context", key: isKey, desc };
    });
    pins.push({ name: "pCTR", kind: "out", no: pins.length + 1, desc: "Output: sigmoid click probability" });
    const half = Math.ceil(pins.length / 2), top = 50, pitch = 26, bodyX = 140, bodyW = 160;
    let h = `<rect x="${bodyX}" y="${top - 22}" width="${bodyW}" height="${half * pitch + 20}" rx="4" class="pkg"/>
      <path d="M${bodyX + bodyW / 2 - 16},${top - 22} a16,16 0 0 0 32,0" fill="var(--paper)" stroke="var(--ink)" stroke-width="1.5"/>
      <circle cx="${bodyX + 16}" cy="${top}" r="4" fill="var(--ink)"/>
      <text x="${bodyX + bodyW / 2}" y="${top + (half / 2 - 1) * pitch}" text-anchor="middle" style="font:800 20px var(--sans)">ARK-3</text>
      <text x="${bodyX + bodyW / 2}" y="${top + (half / 2 - 1) * pitch + 18}" text-anchor="middle" class="pnum" style="font-size:11px">DeepFM</text>
      <text x="${bodyX + bodyW / 2}" y="${top + (half / 2 + 1) * pitch}" text-anchor="middle" class="pnum" style="font-size:10px;letter-spacing:.1em">TOP VIEW</text>`;
    pins.forEach((p) => {
      const left = p.no <= half;
      const row = left ? p.no - 1 : pins.length - p.no;
      const y = top + row * pitch;
      const cls = p.kind === "out" ? "pin out" : p.side === "item" ? "pin item" : "pin";
      const px = left ? bodyX - 34 : bodyX + bodyW;
      h += `<g class="p" data-no="${p.no}">
        <rect x="${px}" y="${y - 6}" width="34" height="12" class="${cls}"/>
        <text x="${left ? bodyX + 8 : bodyX + bodyW - 8}" y="${y + 4}" class="pnum" text-anchor="${left ? "start" : "end"}">${p.no}</text>
        <text x="${left ? px - 6 : px + 40}" y="${y + 4}" class="pname" text-anchor="${left ? "end" : "start"}" ${p.side === "item" ? 'style="fill:var(--accent)"' : ""}>${p.name}</text>
      </g>`;
    });
    $("#pinout").innerHTML = h;
    const tb = $("#pintable tbody");
    tb.innerHTML = pins.map((p) => `<tr data-no="${p.no}"><td class="c">${p.no}</td><td><b ${p.side === "item" ? 'class="hl"' : ""}>${p.name}</b></td><td class="c">${p.kind === "out" ? "O" : "I"}</td><td>${p.kind === "out" ? "—" : p.side === "item" ? "Ad" + (p.key ? " (key)" : "") : "Context"}</td><td>${p.desc}</td></tr>`).join("");
    const hot = (no, on) => $$(`[data-no="${no}"]`).forEach((n) => n.classList.toggle("is-hot", on));
    $$("g.p", $("#pinout")).forEach((g) => {
      const no = +g.dataset.no, p = pins[no - 1];
      g.addEventListener("mouseenter", () => {
        hot(no, true);
        const row = $(`#pintable tr[data-no="${no}"]`), wrap = row.closest(".tbl-wrap");
        wrap.scrollTop = row.offsetTop - wrap.clientHeight / 2;
      });
      g.addEventListener("mousemove", (ev) => showTip(`<b>Pin ${no} · ${p.name}</b><div>${p.desc}</div>`, ev));
      g.addEventListener("mouseleave", () => { hot(no, false); hideTip(); });
    });
    $$("tr[data-no]", tb).forEach((tr) => {
      tr.addEventListener("mouseenter", () => hot(+tr.dataset.no, true));
      tr.addEventListener("mouseleave", () => hot(+tr.dataset.no, false));
    });
  }
})();
