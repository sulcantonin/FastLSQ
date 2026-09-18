/* fastlsq site — no framework, no build step.
   Every figure solves a real least-squares system in the browser. */
(() => {
  "use strict";

  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

  /* ---------------- tiny dense linear algebra ---------------- */

  // Cholesky solve of (AᵀA + μI) β = Aᵀb.  The design matrix is small here
  // (N ≤ 140), so normal equations with a ridge are fine and fast enough to
  // rerun on every slider frame.
  function lstsq(A, b, mu = 1e-10) {
    const m = A.length, n = A[0].length;
    const G = Array.from({ length: n }, () => new Float64Array(n));
    const r = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      for (let j = i; j < n; j++) {
        let s = 0;
        for (let k = 0; k < m; k++) s += A[k][i] * A[k][j];
        G[i][j] = G[j][i] = s;
      }
      G[i][i] += mu;
      let s = 0;
      for (let k = 0; k < m; k++) s += A[k][i] * b[k];
      r[i] = s;
    }
    // Cholesky, with a ridge bump if the factorisation stalls
    const L = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let s = G[i][j];
        for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
        if (i === j) {
          if (s <= 1e-14) s = 1e-14;
          L[i][i] = Math.sqrt(s);
        } else {
          L[i][j] = s / L[j][j];
        }
      }
    }
    const y = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let s = r[i];
      for (let k = 0; k < i; k++) s -= L[i][k] * y[k];
      y[i] = s / L[i][i];
    }
    const x = new Float64Array(n);
    for (let i = n - 1; i >= 0; i--) {
      let s = y[i];
      for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k];
      x[i] = s / L[i][i];
    }
    return x;
  }

  // deterministic PRNG so the page looks the same on every load
  function rng(seed) {
    let s = seed >>> 0;
    return () => {
      s = (s * 1664525 + 1013904223) >>> 0;
      return s / 4294967296;
    };
  }
  const gauss = (r) => {
    const u = Math.max(r(), 1e-12), v = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };

  /* ---------------- canvas helpers ---------------- */

  function setupCanvas(cv) {
    const dpr = window.devicePixelRatio || 1;
    const w = cv.clientWidth, h = Math.round(w * (cv.height / cv.width));
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(h * dpr);
    cv.style.height = h + "px";
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h };
  }

  function axes(ctx, w, h, pad) {
    ctx.strokeStyle = css("--rule");
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i <= 4; i++) {
      const y = pad + (i * (h - 2 * pad)) / 4;
      ctx.moveTo(pad, Math.round(y) + 0.5);
      ctx.lineTo(w - pad, Math.round(y) + 0.5);
    }
    ctx.stroke();
  }

  const plot = (ctx, xs, ys, X, Y, style) => {
    ctx.save();
    Object.assign(ctx, style.ctx || {});
    ctx.strokeStyle = style.color;
    ctx.lineWidth = style.width || 1.6;
    if (style.dash) ctx.setLineDash(style.dash);
    ctx.beginPath();
    for (let i = 0; i < xs.length; i++) {
      const px = X(xs[i]), py = Y(ys[i]);
      i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
    }
    ctx.stroke();
    ctx.restore();
  };

  /* ================= § 01 — live fit ================= */

  const TARGETS = [
    { name: "smooth", f: (x) => Math.sin(3 * x) + 0.4 * Math.cos(7 * x) },
    { name: "kinked", f: (x) => Math.abs(x - 0.3) - 0.4 + 0.3 * Math.sin(4 * x) },
    { name: "chirp",  f: (x) => Math.sin(2 * x * x + 2 * x) },
  ];

  const fitCv = $("#fit-canvas");
  if (fitCv) {
    const M = 220;
    const xs = Array.from({ length: M }, (_, i) => -2 + (4 * i) / (M - 1));
    let target = 0;

    const draw = () => {
      const N = +$("#fit-slider").value;
      const sigma = +$("#sig-slider").value / 10;
      const f = TARGETS[target].f;
      const y = xs.map(f);

      const r = rng(12345);
      const W = Array.from({ length: N }, () => gauss(r) * sigma);
      const B = Array.from({ length: N }, () => r() * 2 * Math.PI);

      const t0 = performance.now();
      const A = xs.map((x) => W.map((w, j) => Math.sin(w * x + B[j]) / Math.sqrt(N)));
      const beta = lstsq(A, y);
      const ms = performance.now() - t0;

      const yhat = A.map((row) => row.reduce((s, v, j) => s + v * beta[j], 0));
      const num = Math.hypot(...yhat.map((v, i) => v - y[i]));
      const den = Math.hypot(...y) || 1;

      const { ctx, w, h } = setupCanvas(fitCv);
      const pad = 26;
      ctx.clearRect(0, 0, w, h);
      axes(ctx, w, h, pad);

      const lo = Math.min(...y, ...yhat), hi = Math.max(...y, ...yhat);
      const sp = (hi - lo) * 0.14 + 1e-9;
      const X = (x) => pad + ((x + 2) / 4) * (w - 2 * pad);
      const Y = (v) => h - pad - ((v - lo + sp) / (hi - lo + 2 * sp)) * (h - 2 * pad);

      // the individual features, very faint — they are texture, not data
      ctx.globalAlpha = 0.5;
      for (let j = 0; j < Math.min(N, 44); j++) {
        const comp = xs.map((x) => (Math.sin(W[j] * x + B[j]) / Math.sqrt(N)) * beta[j]);
        plot(ctx, xs, comp, X, Y, { color: css("--rule-firm"), width: 0.7 });
      }
      ctx.globalAlpha = 1;

      plot(ctx, xs, y, X, Y, { color: css("--ink"), width: 1.5, dash: [5, 4] });
      plot(ctx, xs, yhat, X, Y, { color: css("--accent"), width: 2.1 });

      $("#fit-n").textContent = N;
      $("#fit-err").textContent = (num / den).toExponential(2);
      $("#fit-ms").textContent = ms.toFixed(1) + " ms";
      $("#fit-sig").textContent = sigma.toFixed(1);
    };

    $("#fit-slider").addEventListener("input", draw);
    $("#sig-slider").addEventListener("input", draw);
    $$("[data-target]").forEach((b) =>
      b.addEventListener("click", () => {
        target = +b.dataset.target;
        $$("[data-target]").forEach((o) => o.setAttribute("aria-pressed", String(o === b)));
        draw();
      })
    );
    window.addEventListener("resize", draw);
    draw();
  }

  /* ================= § 03 — derivatives ================= */

  const dCv = $("#deriv-canvas");
  if (dCv) {
    const M = 240, N = 90, sigma = 2.2;
    const xs = Array.from({ length: M }, (_, i) => -2 + (4 * i) / (M - 1));
    const f = (x) => Math.sin(1.7 * x) * Math.exp(-0.12 * x * x);

    // true derivatives by high-order central differences on a fine grid
    const trueD = (k) => {
      const hh = 2e-2;
      const d = (g, x) => (g(x + hh) - g(x - hh)) / (2 * hh);
      let g = f;
      for (let i = 0; i < k; i++) { const p = g; g = (x) => d(p, x); }
      return xs.map(g);
    };

    const r = rng(777);
    const W = Array.from({ length: N }, () => gauss(r) * sigma);
    const B = Array.from({ length: N }, () => r() * 2 * Math.PI);
    const A = xs.map((x) => W.map((w, j) => Math.sin(w * x + B[j]) / Math.sqrt(N)));
    const beta = lstsq(A, xs.map(f), 1e-9);   // solved ONCE, at order 0

    let k = 0, timer = null;

    const draw = () => {
      // closed form: k-th derivative multiplies by W^k and shifts phase by kπ/2
      const yhat = xs.map((x) =>
        W.reduce((s, w, j) =>
          s + beta[j] * Math.pow(w, k) * Math.sin(w * x + B[j] + (k * Math.PI) / 2) / Math.sqrt(N), 0));
      const yt = trueD(k);

      const { ctx, w, h } = setupCanvas(dCv);
      const pad = 26;
      ctx.clearRect(0, 0, w, h);
      axes(ctx, w, h, pad);

      const all = yhat.concat(yt);
      const lo = Math.min(...all), hi = Math.max(...all);
      const sp = (hi - lo) * 0.14 + 1e-9;
      const X = (x) => pad + ((x + 2) / 4) * (w - 2 * pad);
      const Y = (v) => h - pad - ((v - lo + sp) / (hi - lo + 2 * sp)) * (h - 2 * pad);

      plot(ctx, xs, yt, X, Y, { color: css("--ink"), width: 1.5, dash: [5, 4] });
      plot(ctx, xs, yhat, X, Y, { color: css("--accent"), width: 2.1 });

      // compare on the interior only: the finite-difference reference degrades
      // at the edges, which would otherwise be read as the surrogate's error
      let mx = 0;
      for (let i = 12; i < M - 12; i++) mx = Math.max(mx, Math.abs(yhat[i] - yt[i]));
      $("#d-k").textContent = k;
      $("#d-err").textContent = mx.toExponential(1);
      $$("[data-k]").forEach((b) => b.setAttribute("aria-pressed", String(+b.dataset.k === k)));
    };

    $$("[data-k]").forEach((b) =>
      b.addEventListener("click", () => { k = +b.dataset.k; stop(); draw(); })
    );
    const stop = () => { if (timer) { clearInterval(timer); timer = null; $("#d-play").textContent = "▶ sweep"; } };
    $("#d-play").addEventListener("click", () => {
      if (timer) return stop();
      $("#d-play").textContent = "❚❚ pause";
      timer = setInterval(() => { k = (k + 1) % 5; draw(); }, 900);
    });
    window.addEventListener("resize", draw);
    draw();
  }

  /* ================= § 04 — inverse-problem recipes ================= */

  const RECIPES = [
    {
      file: "source_localisation.py",
      unknown: "where the sources are, and how strong (24 parameters)",
      obs: "4 sensors × 60 time samples = 240 numbers",
      fwd: "space–time heat equation, u<sub>t</sub> − αΔu = Σ f<sub>k</sub>",
      gives: "∂<sub>t</sub> and Δ in closed form, so the operator assembles once and factors once",
      note: "<b>The factor is the whole trick.</b> A thousand L-BFGS-B iterations would mean a thousand PDE solves. Here it means a thousand back-substitutions against one Cholesky factor.",
      code: `L = Op.dt(d=3) - alpha * Op.laplacian(d=3, dims=(0, 1))
A = L.apply(basis, x)          # assembled ONCE
c = cho_factor(A.T @ A + mu * I)

def forward(theta):            # called ~1000x by the optimiser
    beta = cho_solve(c, A.T @ b(theta))
    return sensors @ beta      # back-substitution only`,
    },
    {
      file: "parameter_id.py",
      unknown: "a PDE coefficient — wavenumber, diffusivity, fractional order s",
      obs: "sparse, noisy measurements of the field",
      fwd: "any operator the coefficient enters linearly or smoothly",
      gives: "coefficients may be <code>nn.Parameter</code>; gradients flow through the solve",
      note: "<b>The coefficient is a tensor, not a loop variable.</b> Put an <code>nn.Parameter</code> into the operator and AdamW optimises it through the prebuilt solve — including the fractional order of <code>(−Δ)^s</code>.",
      code: `k = torch.nn.Parameter(torch.tensor(3.0))

def loss():
    L = Op.laplacian(d=2) + k**2 * Op.identity(d=2)
    A = L.apply(basis, x)      # k is live in the graph
    beta = solve_lstsq(A, b)
    return ((obs_op @ beta - obs)**2).mean()

torch.optim.AdamW([k], lr=1e-2)`,
    },
    {
      file: "tomography.py",
      unknown: "a field seen only through line integrals",
      obs: "projections ∫ f δ(c·z − u) dz at several angles",
      fwd: "ProjectionOperator on a GaussianWindowedBasis",
      gives: "the hyperplane integral of a Gaussian × plane wave is analytic — no quadrature",
      note: "<b>Quadrature-free, and differentiable in the optics.</b> The projection direction <code>c</code> stays in the graph, so the acquisition geometry itself can be optimised — experiment design, not just reconstruction.",
      code: `basis = GaussianWindowedBasis.random(2, 1200, sigma=6.0)
P     = ProjectionOperator.from_transport(M, e)

A = torch.cat([P.apply(basis, u_k, c=c_k)
               for c_k, u_k in angles])
beta = solve_lstsq(A, torch.cat(projections))

f_hat = lambda z: basis.evaluate(z) @ beta`,
    },
    {
      file: "memory_kernel.py",
      unknown: "the strength — or the shape — of a memory kernel",
      obs: "a noisy trajectory of the response",
      fwd: "Volterra / Fredholm integro-differential equation",
      gives: "∫ and ∂ assemble into the <em>same</em> least-squares block",
      note: "<b>Integrals cost what derivatives cost.</b> A plane wave integrates as easily as it differentiates, so a memory term is another matrix block — not a quadrature rule bolted onto a time-stepper.",
      code: `lam = torch.nn.Parameter(torch.tensor(0.5))

L = Op.d(0, d=1) + lam * IntegralOperator.volterra(lo=0.0)
A = L.apply(basis, x)          # ∂ and ∫ in one block
beta = solve_lstsq(A, f(x))

loss = ((basis.evaluate(t_obs) @ beta - y_obs)**2).mean()`,
    },
  ];

  const invSeg = $("#inv-seg");
  if (invSeg) {
    const show = (i) => {
      const r = RECIPES[i];
      $("#inv-file").textContent = r.file;
      $("#inv-unknown").innerHTML = r.unknown;
      $("#inv-obs").innerHTML = r.obs;
      $("#inv-fwd").innerHTML = r.fwd;
      $("#inv-gives").innerHTML = r.gives;
      $("#inv-note").innerHTML = r.note;
      $("#inv-code").textContent = r.code;
      $$("[data-inv]").forEach((b) => b.setAttribute("aria-pressed", String(+b.dataset.inv === i)));
    };
    $$("[data-inv]").forEach((b) => b.addEventListener("click", () => show(+b.dataset.inv)));
    show(0);
  }

  /* ================= copy buttons ================= */

  $$("[data-copy]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const el = $(btn.dataset.copy);
      if (!el) return;
      try {
        await navigator.clipboard.writeText(el.textContent.trim());
        const was = btn.textContent;
        btn.textContent = "copied";
        setTimeout(() => (btn.textContent = was), 1400);
      } catch { /* clipboard blocked — leave the text selectable */ }
    });
  });

  /* ================= search ================= */

  // Build the index from the DOM, so it can never drift from the page.
  // <br> is a word boundary: without this, "any operator.<br>One formula" indexes
  // and renders as "operator.One".
  const flat = (el) => {
    const c = el.cloneNode(true);
    c.querySelectorAll("br").forEach((b) => b.replaceWith(" "));
    return c.textContent.replace(/\s+/g, " ").trim();
  };

  const INDEX = $$("main section").flatMap((sec) => {
    const secName = (sec.querySelector(".mark")?.textContent || "").replace(/§\s*\d+\s*—\s*/, "").trim();
    return $$("h2, h3, p, td, li", sec)
      .map((el) => ({ el, text: flat(el) }))
      .filter((r) => r.text.length > 24)
      .map((r) => ({
        id: sec.id,
        section: secName || sec.id,
        title: (r.el.tagName === "H2" || r.el.tagName === "H3")
          ? r.text
          : (sec.querySelector("h2") ? flat(sec.querySelector("h2")) : sec.id),
        text: r.text,
      }));
  });

  const dlg = $("#search"), input = $("#search-input"), out = $("#search-results");

  const esc = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const render = (q) => {
    out.innerHTML = "";
    const terms = q.toLowerCase().split(/\s+/).filter(Boolean);
    if (!terms.length) {
      out.innerHTML = `<li class="search-empty">Type to search headings, prose and tables.</li>`;
      return;
    }
    const hits = INDEX
      .map((r) => {
        const hay = (r.title + " " + r.text).toLowerCase();
        if (!terms.every((t) => hay.includes(t))) return null;
        // earliest match ranks higher; heading matches rank higher still
        const pos = hay.indexOf(terms[0]);
        return { r, score: pos + (r.title.toLowerCase().includes(terms[0]) ? -400 : 0) };
      })
      .filter(Boolean)
      .sort((a, b) => a.score - b.score)
      .slice(0, 12);

    if (!hits.length) {
      out.innerHTML = `<li class="search-empty">Nothing on this page matches “${q}”.</li>`;
      return;
    }
    const re = new RegExp("(" + terms.map(esc).join("|") + ")", "ig");
    for (const { r } of hits) {
      const i = r.text.toLowerCase().indexOf(terms[0]);
      const from = Math.max(0, i - 60);
      const snip = (from ? "…" : "") + r.text.slice(from, from + 170) + (r.text.length > from + 170 ? "…" : "");
      const li = document.createElement("li");
      li.innerHTML = `<a href="#${r.id}">
          <span class="r-s">${r.section}</span>
          <span class="r-t">${r.title}</span>
          <span class="r-x">${snip.replace(re, "<mark>$1</mark>")}</span>
        </a>`;
      li.querySelector("a").addEventListener("click", () => dlg.close());
      out.appendChild(li);
    }
  };

  const openSearch = () => {
    if (!dlg.open) dlg.showModal();
    input.value = "";
    render("");
    input.focus();
  };
  $("#open-search")?.addEventListener("click", openSearch);
  input?.addEventListener("input", () => render(input.value));
  document.addEventListener("keydown", (e) => {
    const typing = /^(INPUT|TEXTAREA)$/.test(document.activeElement?.tagName || "");
    if (e.key === "/" && !typing) { e.preventDefault(); openSearch(); }
    if ((e.key === "k" || e.key === "K") && (e.metaKey || e.ctrlKey)) { e.preventDefault(); openSearch(); }
  });
  out?.addEventListener("keydown", (e) => { if (e.key === "Escape") dlg.close(); });
})();
