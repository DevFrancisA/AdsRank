/* ARK-3 evaluation board: the trained funnel, live, in onnxruntime-web.
 * query tower (ONNX) -> exact inner product over item vectors -> top-K ->
 * DeepFM (ONNX) on the K pairs -> top-10, compared with DeepFM over the
 * whole catalog. */
(function () {
  const $ = (s) => document.querySelector(s);
  const led = $("#evm-led"), bar = $("#evm-bar"), runBtn = $("#btn-run");
  const st = { demo: null, vecs: null, ranker: null, tower: null, q: 0, masked: new Set(), busy: false };
  const CHUNK = 2000;

  function setLed(cls, label) { led.className = "led " + cls; led.textContent = label; }
  const progress = (v) => (bar.style.width = (100 * v).toFixed(1) + "%");
  const nextFrame = () => new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));

  async function boot() {
    if (!window.ort) { setLed("err", "onnxruntime-web failed to load"); return; }
    setLed("busy", "loading models");
    try {
      ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
      ort.env.wasm.numThreads = 1;
      const [demo, vecBuf, ranker, tower] = await Promise.all([
        fetch("assets/demo.json").then((r) => r.json()),
        fetch("assets/item_vectors.bin").then((r) => r.arrayBuffer()),
        ort.InferenceSession.create("assets/ranker_web.onnx", { executionProviders: ["wasm"] }),
        ort.InferenceSession.create("assets/query_tower_web.onnx", { executionProviders: ["wasm"] }),
      ]);
      Object.assign(st, { demo, vecs: new Float32Array(vecBuf), ranker, tower });
    } catch (e) {
      console.error(e);
      setLed("err", "load failed");
      return;
    }
    const sel = $("#q-sel");
    sel.innerHTML = st.demo.queries.map((q) => `<option value="${q.id}">test #${String(q.id).padStart(3, "0")}</option>`).join("");
    sel.addEventListener("change", () => { st.q = +sel.value; st.masked.clear(); drawPins(); run(); });
    const rng = $("#k-rng");
    rng.value = st.demo.k_default;
    $("#k-val").textContent = rng.value;
    rng.addEventListener("input", () => ($("#k-val").textContent = rng.value));
    rng.addEventListener("change", run);
    $("#btn-reset").addEventListener("click", () => { st.masked.clear(); drawPins(); run(); });
    runBtn.addEventListener("click", run);
    runBtn.disabled = false;
    drawPins();
    setLed("on", "ready");
    run();
  }

  function drawPins() {
    const d = st.demo, q = d.queries[st.q];
    $("#pins").innerHTML = d.ctx_idx.map((j, c) => {
      const m = st.masked.has(c);
      return `<button type="button" class="hp${m ? " masked" : ""}" data-c="${c}" title="${d.fields[j]} = ${q.tokens[c]}${m ? " (forced missing)" : ""}"><b>${d.fields[j]}</b><span>${q.tokens[c]}</span></button>`;
    }).join("") + d.item_fields.map((n) => `<span class="hp item" title="ad-side pin: supplied by each candidate"><b>${n}</b><span>per ad</span></span>`).join("");
    document.querySelectorAll("#pins button.hp").forEach((b) => b.addEventListener("click", () => {
      const c = +b.dataset.c;
      st.masked.has(c) ? st.masked.delete(c) : st.masked.add(c);
      drawPins();
      run();
    }));
  }

  function contextRow() {
    const d = st.demo, q = d.queries[st.q];
    return q.ctx.map((v, c) => (st.masked.has(c) ? d.missing[d.ctx_idx[c]] : v));
  }

  // DeepFM over a list of item ids for the current context; chunked so the UI breathes
  async function rank(ids, onProgress) {
    const d = st.demo, F = d.fields.length, nI = d.item_idx.length;
    const ctx = contextRow();
    const out = new Float32Array(ids.length);
    for (let s = 0; s < ids.length; s += CHUNK) {
      const n = Math.min(CHUNK, ids.length - s);
      const x = new BigInt64Array(n * F);
      for (let r = 0; r < n; r++) {
        const base = r * F, it = ids[s + r] * nI;
        for (let c = 0; c < ctx.length; c++) x[base + d.ctx_idx[c]] = BigInt(ctx[c]);
        for (let c = 0; c < nI; c++) x[base + d.item_idx[c]] = BigInt(d.items[it + c]);
      }
      const res = await st.ranker.run({ x: new ort.Tensor("int64", x, [n, F]) });
      out.set(res.y.data, s);
      if (onProgress) { onProgress((s + n) / ids.length); await nextFrame(); }
    }
    return out;
  }

  function topN(ids, scores, n) {
    const idx = Array.from(scores.keys()).sort((a, b) => scores[b] - scores[a]).slice(0, n);
    return idx.map((i) => ({ item: ids[i], p: scores[i] }));
  }

  async function run() {
    if (st.busy || !st.demo) return;
    st.busy = true;
    runBtn.disabled = true;
    setLed("busy", "running");
    progress(0);
    const d = st.demo, K = +$("#k-rng").value, N = d.n_items, D = d.vec_dim;
    ["#list-f", "#list-e"].forEach((s) => $(s).classList.add("pending"));
    try {
      // stage 1+2: query tower, exact inner-product shortlist
      const t0 = performance.now();
      const ctx = contextRow();
      const tq = await st.tower.run({ x: new ort.Tensor("int64", BigInt64Array.from(ctx.map(BigInt)), [1, ctx.length]) });
      const qv = tq.y.data;
      const sc = new Float32Array(N);
      for (let i = 0; i < N; i++) {
        let s = st.vecs[i * D + D - 1];
        for (let k = 0; k < D - 1; k++) s += qv[k] * st.vecs[i * D + k];
        sc[i] = s;
      }
      const cand = Array.from(sc.keys()).sort((a, b) => sc[b] - sc[a]).slice(0, K);
      // stage 3: DeepFM on the shortlist
      const ps = await rank(cand, (v) => progress(0.1 * v));
      const funnel = topN(cand, ps, 10);
      const tf = performance.now() - t0;
      $("#ro-tf").innerHTML = `${tf.toFixed(0)}<small>ms</small>`;
      $("#ro-calls").innerHTML = `${K.toLocaleString()}`;
      $("#ro-calls-s").textContent = `vs. ${N.toLocaleString()} (${(100 * (1 - K / N)).toFixed(0)}% fewer)`;
      renderList("#list-f", funnel, null);

      // oracle: DeepFM over the entire catalog
      const t1 = performance.now();
      const all = Array.from({ length: N }, (_, i) => i);
      const pa = await rank(all, (v) => progress(0.1 + 0.9 * v));
      const oracle = topN(all, pa, 10);
      const te = performance.now() - t1;
      const oset = new Set(oracle.map((o) => o.item)), fset = new Set(funnel.map((o) => o.item));
      const hits = funnel.filter((o) => oset.has(o.item)).length;
      $("#ro-te").innerHTML = `${te.toFixed(0)}<small>ms</small>`;
      $("#ro-ret").innerHTML = `${hits}<small>/10</small>`;
      renderList("#list-f", funnel, oset);
      renderList("#list-e", oracle, fset, new Map(cand.map((c, i) => [c, i])));
      setLed("on", `done · ${st.masked.size} pin${st.masked.size === 1 ? "" : "s"} masked`);
    } catch (e) {
      console.error(e);
      setLed("err", "inference error");
    }
    progress(1);
    st.busy = false;
    runBtn.disabled = false;
  }

  function renderList(sel, rows, other, candRank) {
    const d = st.demo, ol = $(sel);
    ol.classList.remove("pending");
    ol.innerHTML = rows.map((r) => {
      const inOther = other ? other.has(r.item) : null;
      const tag = other === null ? "" : sel === "#list-f"
        ? `<span class="m">${inOther ? "IN ORACLE" : "MISS"}</span>`
        : `<span class="m">${inOther ? "FOUND" : candRank && candRank.has(r.item) ? "RANK " + (candRank.get(r.item) + 1) : "NOT RETRIEVED"}</span>`;
      const cls = other === null ? "" : inOther ? "hit" : "miss";
      return `<li class="${cls}" title="${d.key_field} = ${d.item_keys[r.item]} · ${d.item_freq[r.item].toLocaleString()} train impressions"><span class="id">AD-${String(r.item + 1).padStart(5, "0")} · ${d.item_keys[r.item]}</span><span class="p">${(100 * r.p).toFixed(2)}%</span>${tag || "<span></span>"}</li>`;
    }).join("");
  }

  // Boot when the board scrolls near the viewport, so the page itself stays light.
  const evm = document.getElementById("evm");
  if ("IntersectionObserver" in window) {
    const io = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) { io.disconnect(); boot(); } }, { rootMargin: "600px" });
    io.observe(evm);
  } else boot();
})();
