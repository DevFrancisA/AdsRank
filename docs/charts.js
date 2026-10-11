/* Datasheet-style SVG charts: hairline graph-paper grid, ink series told apart
 * by dash pattern + marker, one accent series. Every chart gets a hover layer
 * and a tabulated-data disclosure. No dependencies. */
(function () {
  const NS = "http://www.w3.org/2000/svg";
  const tip = () => document.getElementById("tip");

  function el(tag, attrs = {}, parent) {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) if (v !== undefined && v !== null) n.setAttribute(k, v);
    if (parent) parent.appendChild(n);
    return n;
  }
  function text(parent, x, y, s, attrs = {}) {
    const t = el("text", { x, y, ...attrs }, parent);
    t.textContent = s;
    return t;
  }

  // Series styles: index 0 is the accent, the rest are ink with distinct dashes.
  const STYLES = [
    { color: "var(--accent)", dash: null, marker: "circle", width: 2 },
    { color: "var(--ink)", dash: "6 3", marker: "square", width: 1.6 },
    { color: "var(--ink-2)", dash: "2 2.5", marker: "triangle", width: 1.6 },
    { color: "var(--ink-3)", dash: "9 3 2 3", marker: "diamond", width: 1.6 },
  ];

  function marker(g, kind, x, y, color, r = 4) {
    const common = { fill: "var(--paper)", stroke: color, "stroke-width": 1.6 };
    if (kind === "circle") return el("circle", { cx: x, cy: y, r, ...common, fill: color, stroke: "var(--paper)", "stroke-width": 1.5 }, g);
    if (kind === "square") return el("rect", { x: x - r + .5, y: y - r + .5, width: 2 * r - 1, height: 2 * r - 1, ...common }, g);
    if (kind === "triangle") return el("path", { d: `M${x},${y - r - .5}L${x + r},${y + r - 1}L${x - r},${y + r - 1}Z`, ...common }, g);
    return el("path", { d: `M${x},${y - r}L${x + r},${y}L${x},${y + r}L${x - r},${y}Z`, ...common }, g);
  }

  function niceTicks(min, max, n = 5) {
    const span = max - min;
    const step0 = Math.pow(10, Math.floor(Math.log10(span / n)));
    const err = span / n / step0;
    const step = step0 * (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1);
    const out = [];
    for (let v = Math.ceil(min / step - 1e-9) * step; v <= max + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }

  function scale(ax, r0, r1) {
    if (ax.log) {
      const a = Math.log10(ax.min), b = Math.log10(ax.max);
      return (v) => r0 + (Math.log10(v) - a) / (b - a) * (r1 - r0);
    }
    return (v) => r0 + (v - ax.min) / (ax.max - ax.min) * (r1 - r0);
  }

  function showTip(html, ev) {
    const t = tip();
    t.innerHTML = html;
    t.style.display = "block";
    const w = t.offsetWidth, h = t.offsetHeight;
    let x = ev.clientX + 14, y = ev.clientY + 14;
    if (x + w > innerWidth - 8) x = ev.clientX - w - 14;
    if (y + h > innerHeight - 8) y = ev.clientY - h - 14;
    t.style.left = x + "px";
    t.style.top = y + "px";
  }
  function hideTip() { tip().style.display = "none"; }

  function figure(container, { id, caption, cond, table }) {
    const fig = document.createElement("figure");
    if (id) fig.id = id;
    const plot = document.createElement("div");
    plot.className = "plot";
    fig.appendChild(plot);
    const cap = document.createElement("figcaption");
    cap.innerHTML = caption + (cond ? `<span class="cond">${cond}</span>` : "");
    fig.appendChild(cap);
    if (table) {
      const d = document.createElement("details");
      d.className = "data";
      d.innerHTML = `<summary>Tabulated data</summary><div class="tbl-wrap"><table><thead><tr>${
        table.head.map((h) => `<th>${h}</th>`).join("")}</tr></thead><tbody>${
        table.rows.map((r) => `<tr>${r.map((c) => `<td class="r">${c}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
      fig.appendChild(d);
    }
    container.appendChild(fig);
    return plot;
  }

  function frame(svg, W, H, m, x, y) {
    const sx = scale(x, m.l, W - m.r), sy = scale(y, H - m.b, m.t);
    const g = el("g", {}, svg);
    const xt = x.ticks || niceTicks(x.min, x.max, x.n || 6);
    const yt = y.ticks || niceTicks(y.min, y.max, y.n || 5);
    // minor grid (datasheet graph paper)
    if (!x.log && xt.length > 1) {
      const st = (xt[1] - xt[0]) / 2;
      for (let v = xt[0] - st; v <= x.max; v += st) if (v >= x.min) el("line", { x1: sx(v), x2: sx(v), y1: m.t, y2: H - m.b, class: "gridline" }, g);
    }
    if (!y.log && yt.length > 1) {
      const st = (yt[1] - yt[0]) / 2;
      for (let v = yt[0] - st; v <= y.max; v += st) if (v >= y.min) el("line", { y1: sy(v), y2: sy(v), x1: m.l, x2: W - m.r, class: "gridline" }, g);
    }
    const auto = (ts) => { const st = ts.length > 1 ? Math.abs(ts[1] - ts[0]) : 1; const d = Math.max(0, -Math.floor(Math.log10(st) + 1e-9)); return (v) => v.toFixed(d); };
    if (!x.fmt) x.fmt = auto(xt);
    if (!y.fmt) y.fmt = auto(yt);
    const tg = el("g", { class: "tick" }, g);
    xt.forEach((v) => {
      el("line", { x1: sx(v), x2: sx(v), y1: m.t, y2: H - m.b, class: "gridline", style: "stroke:var(--hair)" }, g);
      text(tg, sx(v), H - m.b + 15, (x.fmt || String)(v), { "text-anchor": "middle" });
    });
    yt.forEach((v) => {
      el("line", { y1: sy(v), y2: sy(v), x1: m.l, x2: W - m.r, class: "gridline", style: "stroke:var(--hair)" }, g);
      text(tg, m.l - 6, sy(v) + 3.5, (y.fmt || String)(v), { "text-anchor": "end" });
    });
    el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, class: "frame" }, g);
    text(g, (m.l + W - m.r) / 2, H - 6, x.label, { "text-anchor": "middle", class: "axis-label" });
    text(g, 0, 0, y.label, { "text-anchor": "middle", class: "axis-label", transform: `translate(13 ${(m.t + H - m.b) / 2}) rotate(-90)` });
    return { sx, sy };
  }

  function condBox(svg, lines, x, y, anchorRight) {
    if (!lines || !lines.length) return;
    const g = el("g", { class: "cond-box" }, svg);
    const w = Math.max(...lines.map((s) => s.length)) * 5.2 + 12, h = lines.length * 13 + 6;
    const x0 = anchorRight ? x - w : x;
    el("rect", { x: x0, y, width: w, height: h }, g);
    lines.forEach((s, i) => text(g, x0 + 6, y + 14 + i * 13, s));
  }

  function legend(svg, series, x, y, anchorRight) {
    const g = el("g", { class: "legend" }, svg);
    const w = Math.max(...series.map((s) => s.name.length)) * 5.6 + 44, h = series.length * 15 + 8;
    const x0 = anchorRight ? x - w : x;
    el("rect", { x: x0, y, width: w, height: h, fill: "var(--paper)", stroke: "var(--ink-3)", "stroke-width": .75 }, g);
    series.forEach((s, i) => {
      const st = s.style, yy = y + 13 + i * 15;
      el("line", { x1: x0 + 7, x2: x0 + 31, y1: yy - 3.5, y2: yy - 3.5, stroke: st.color, "stroke-width": st.width, "stroke-dasharray": st.dash }, g);
      if (st.marker) marker(g, st.marker, x0 + 19, yy - 3.5, st.color, 3.2);
      text(g, x0 + 37, yy, s.name);
    });
  }

  /* line(container, {x, y, series:[{name, pts:[[x,y]...], style?, label?}], refs, caption, cond, condLines, legendAt, fmtTip}) */
  function line(container, o) {
    const W = 520, H = 330, m = { l: 54, r: 16, t: 14, b: 42, ...(o.margin || {}) };
    const series = o.series.map((s, i) => ({ ...s, style: { ...STYLES[s.styleIdx ?? i], ...(s.style || {}) } }));
    const plot = figure(container, o);
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.caption.replace(/<[^>]+>/g, "") }, plot);
    const { sx, sy } = frame(svg, W, H, m, o.x, o.y);
    const clip = "c" + Math.random().toString(36).slice(2);
    el("clipPath", { id: clip }, el("defs", {}, svg)).appendChild(el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b }));
    const body = el("g", { "clip-path": `url(#${clip})` }, svg);
    (o.refs || []).forEach((r) => {
      if (r.y !== undefined) {
        el("line", { x1: m.l, x2: W - m.r, y1: sy(r.y), y2: sy(r.y), stroke: "var(--ink)", "stroke-width": 1, "stroke-dasharray": "4 3" }, body);
        text(svg, r.right ? W - m.r - 4 : m.l + 4, sy(r.y) - 4, r.label, { class: "direct", "text-anchor": r.right ? "end" : "start", style: "font-weight:600" });
      }
      if (r.x !== undefined) {
        el("line", { y1: m.t, y2: H - m.b, x1: sx(r.x), x2: sx(r.x), stroke: "var(--ink)", "stroke-width": 1, "stroke-dasharray": "4 3" }, body);
        text(svg, sx(r.x) + 4, m.t + 12, r.label, { class: "direct", style: "font-weight:600" });
      }
    });
    if (o.diagonal) el("line", { x1: sx(o.x.min), y1: sy(o.x.min), x2: sx(o.x.max), y2: sy(o.x.max), stroke: "var(--ink-3)", "stroke-width": 1 }, body);
    // draw back-to-front so the accent series sits on top
    [...series].reverse().forEach((s) => {
      const st = s.style;
      if (!o.noLine) {
        const d = s.pts.map((p, i) => `${i ? "L" : "M"}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join("");
        el("path", { d, fill: "none", stroke: st.color, "stroke-width": st.width, "stroke-dasharray": st.dash, "stroke-linejoin": "round" }, body);
      }
      if (st.marker && s.pts.length <= 40) {
        const mg = el("g", {}, svg);
        s.pts.forEach((p) => marker(mg, st.marker, sx(p[0]), sy(p[1]), st.color));
      }
      if (s.label) {
        const p = s.label.at === "first" ? s.pts[0] : s.pts[s.pts.length - 1];
        text(svg, sx(p[0]) + (s.label.dx || 6), sy(p[1]) + (s.label.dy || 4), s.label.text || s.name,
          { class: "direct", "text-anchor": s.label.anchor || "start", style: `fill:${st.color === "var(--accent)" ? "var(--accent)" : "var(--ink)"}` });
      }
    });
    if (o.legendAt !== false && series.length > 1) {
      const la = o.legendAt || "br";
      if (typeof la === "object") legend(svg, series, m.l + la.x * (W - m.l - m.r), m.t + la.y * (H - m.t - m.b), false);
      else legend(svg, series, la.includes("r") ? W - m.r - 8 : m.l + 8, la.includes("t") ? m.t + 8 : H - m.b - 8 - (series.length * 15 + 8), la.includes("r"));
    }
    if (o.condLines) {
      const ca = o.condAt || "tl";
      condBox(svg, o.condLines, ca.includes("r") ? W - m.r - 8 : m.l + 8, ca.includes("t") ? m.t + 8 : H - m.b - 8 - (o.condLines.length * 13 + 6), ca.includes("r"));
    }
    // hover: crosshair snapped to nearest x across all series
    const xs = [...new Set(series.flatMap((s) => s.pts.map((p) => p[0])))].sort((a, b) => a - b);
    const ch = el("line", { y1: m.t, y2: H - m.b, class: "crosshair", visibility: "hidden" }, svg);
    const hits = el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" }, svg);
    hits.addEventListener("mousemove", (ev) => {
      const pt = svg.createSVGPoint();
      pt.x = ev.clientX; pt.y = ev.clientY;
      const loc = pt.matrixTransform(svg.getScreenCTM().inverse());
      let best = xs[0];
      xs.forEach((v) => { if (Math.abs(sx(v) - loc.x) < Math.abs(sx(best) - loc.x)) best = v; });
      ch.setAttribute("x1", sx(best)); ch.setAttribute("x2", sx(best)); ch.setAttribute("visibility", "visible");
      const rows = series.map((s) => {
        const p = s.pts.find((q) => q[0] === best);
        if (!p) return "";
        return `<div class="row"><span><span class="sw" style="border-color:${s.style.color};border-top-style:${s.style.dash ? "dashed" : "solid"}"></span>${s.name}</span><b>${(o.y.tipFmt || o.y.fmt || String)(p[1])}</b></div>`;
      }).join("");
      showTip(`<b>${o.x.tipLabel || o.x.label}: ${(o.x.tipFmt || o.x.fmt || String)(best)}</b>${rows}`, ev);
    });
    hits.addEventListener("mouseleave", () => { ch.setAttribute("visibility", "hidden"); hideTip(); });
    return svg;
  }

  /* range dot plot: rows of {name, p50, p95, p99, accent} on one horizontal value axis */
  function ranges(container, o) {
    const rowH = 34, W = 520, m = { l: 128, r: 18, t: 14, b: 42 };
    const H = m.t + m.b + o.rows.length * rowH;
    const plot = figure(container, o);
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.caption.replace(/<[^>]+>/g, "") }, plot);
    const x = o.x;
    const sx = scale(x, m.l, W - m.r);
    const xt = x.ticks || niceTicks(x.min, x.max, 6);
    xt.forEach((v) => {
      el("line", { x1: sx(v), x2: sx(v), y1: m.t, y2: H - m.b, class: "gridline", style: "stroke:var(--hair)" }, svg);
      text(el("g", { class: "tick" }, svg), sx(v), H - m.b + 15, (x.fmt || String)(v), { "text-anchor": "middle" });
    });
    el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, class: "frame" }, svg);
    text(svg, (m.l + W - m.r) / 2, H - 6, x.label, { "text-anchor": "middle", class: "axis-label" });
    o.rows.forEach((r, i) => {
      const y = m.t + rowH * (i + .5);
      const c = r.accent ? "var(--accent)" : "var(--ink)";
      text(svg, m.l - 8, y + 4, r.name, { "text-anchor": "end", style: `font-size:12px;font-weight:${r.accent ? 700 : 500}` });
      el("line", { x1: sx(r.p50), x2: sx(r.p99), y1: y, y2: y, stroke: c, "stroke-width": 2 }, svg);
      el("line", { x1: sx(r.p99), x2: sx(r.p99), y1: y - 5, y2: y + 5, stroke: c, "stroke-width": 1.5 }, svg);
      el("circle", { cx: sx(r.p50), cy: y, r: 4, fill: "var(--paper)", stroke: c, "stroke-width": 1.8 }, svg);
      el("rect", { x: sx(r.p95) - 4, y: y - 4, width: 8, height: 8, fill: c, stroke: "var(--paper)", "stroke-width": 1.5 }, svg);
      text(svg, Math.min(sx(r.p99) + 7, W - m.r - 4), y + 4, (x.labelFmt || x.fmt || String)(r.p95) + " p95",
        { class: "direct", style: `fill:${c}`, "text-anchor": sx(r.p99) + 60 > W - m.r ? "end" : "start", dy: sx(r.p99) + 60 > W - m.r ? -9 : 0 });
      const hit = el("rect", { x: m.l, y: y - rowH / 2, width: W - m.l - m.r, height: rowH, fill: "transparent" }, svg);
      hit.addEventListener("mousemove", (ev) => showTip(
        `<b>${r.name}</b>${r.sub ? `<div>${r.sub}</div>` : ""}<div class="row"><span>p50</span><b>${r.p50.toFixed(2)} ms</b></div><div class="row"><span>p95</span><b>${r.p95.toFixed(2)} ms</b></div><div class="row"><span>p99</span><b>${r.p99.toFixed(2)} ms</b></div>`, ev));
      hit.addEventListener("mouseleave", hideTip);
    });
    // key: open circle p50, filled square p95, end tick p99
    const lg = el("g", { class: "legend" }, svg);
    const ly = H - 8, lx = 6;
    el("circle", { cx: lx + 4, cy: ly - 4, r: 3.5, fill: "var(--paper)", stroke: "var(--ink)", "stroke-width": 1.5 }, lg);
    text(lg, lx + 11, ly, "p50");
    el("rect", { x: lx + 34, y: ly - 7.5, width: 7, height: 7, fill: "var(--ink)" }, lg);
    text(lg, lx + 45, ly, "p95");
    el("line", { x1: lx + 70, x2: lx + 70, y1: ly - 9, y2: ly + 1, stroke: "var(--ink)", "stroke-width": 1.5 }, lg);
    text(lg, lx + 75, ly, "p99");
    return svg;
  }

  /* heatmap of a square matrix, single-hue ramp (ink opacity = dependence) */
  function heat(container, o) {
    const n = o.labels.length, cell = 15, m = { l: 40, t: 40, r: 70, b: 10 };
    const W = m.l + n * cell + m.r, H = m.t + n * cell + m.b;
    const plot = figure(container, o);
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.caption.replace(/<[^>]+>/g, "") }, plot);
    const g = el("g", {}, svg);
    o.labels.forEach((lab, i) => {
      const hot = o.hot && o.hot.includes(lab);
      const st = `font-size:9.5px;${hot ? "fill:var(--accent);font-weight:700" : "fill:var(--ink-2)"}`;
      text(svg, m.l - 4, m.t + i * cell + cell * .7, lab, { "text-anchor": "end", style: st });
      text(svg, 0, 0, lab, { style: st, transform: `translate(${m.l + i * cell + cell * .68} ${m.t - 4}) rotate(-90)` });
    });
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const u = o.matrix[i][j];
      const strength = i === j ? 0 : Math.max(0, 1 - u);
      const r = el("rect", {
        x: m.l + j * cell + 1, y: m.t + i * cell + 1, width: cell - 2, height: cell - 2,
        fill: i === j ? "var(--fill)" : (o.hot && o.hot.includes(o.labels[i]) && o.hot.includes(o.labels[j]) ? "var(--accent)" : "var(--ink)"),
        "fill-opacity": i === j ? 1 : (0.06 + 0.94 * Math.pow(strength, 2.2)).toFixed(3),
      }, g);
      if (i !== j) {
        r.addEventListener("mousemove", (ev) => showTip(`<b>${o.labels[i]} → ${o.labels[j]}</b><div class="row"><span>U(${o.labels[j]} | ${o.labels[i]})</span><b>${u.toFixed(3)}</b></div><div>${u < o.threshold ? `${o.labels[i]} (almost) determines ${o.labels[j]}` : "weak dependence"}</div>`, ev));
        r.addEventListener("mouseleave", hideTip);
      }
    }
    // scale legend
    const lx = m.l + n * cell + 18, ly = m.t;
    text(svg, lx, ly - 6, "1 − U", { style: "font-size:10px;font-weight:700" });
    for (let k = 0; k <= 10; k++) {
      const s = 1 - k / 10;
      el("rect", { x: lx, y: ly + k * 14, width: 14, height: 13, fill: "var(--ink)", "fill-opacity": (0.06 + 0.94 * Math.pow(s, 2.2)).toFixed(3) }, svg);
      if (k % 5 === 0) text(svg, lx + 19, ly + k * 14 + 10, s.toFixed(1), { style: "font-size:9.5px;fill:var(--ink-2)" });
    }
    text(svg, lx, ly + 11 * 14 + 14, "row K", { style: "font-size:9.5px;fill:var(--ink-2)" });
    text(svg, lx, ly + 11 * 14 + 26, "determines", { style: "font-size:9.5px;fill:var(--ink-2)" });
    text(svg, lx, ly + 11 * 14 + 38, "column J", { style: "font-size:9.5px;fill:var(--ink-2)" });
    return svg;
  }

  window.Charts = { line, ranges, heat, el, text, marker, showTip, hideTip, STYLES };
})();
