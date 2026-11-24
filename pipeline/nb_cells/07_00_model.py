# ============================================================================
# CELL 7 - Model, losses, and the two fixes to the fusion/gating path
#
# This is your working 2.5-D MW-HQNet, kept as close to the executed code as possible so the
# numbers stay comparable, with four changes:
#   1. `fuse()` renormalises the categorical probabilities after blending (action-plan Sec.4.2).
#   2. the kidney gate in post-processing is built from the PREDICTED organ mask and admits
#      predicted lesions, instead of only voxels whose argmax is kidney (action-plan Sec.4.3).
#   3. a WIDTH_MULT knob gives a capacity-matched baseline for reviewer Sec.1.5.
#   4. a MAC/parameter counter so Table V can report computational cost honestly.
# ============================================================================
MODEL_NEEDED = ENABLED["train"] or ENABLED["infer"] or ENABLED["run_all"]

if MODEL_NEEDED:
    import random as _random
    torch = ensure_torch()
    timm = ensure_timm()
    import torch.nn as nn
    import torch.nn.functional as F

    # ------------------------------------------------------------------ model cfg
    # Build on the FULL run CFG (seeds, budgets, thresholds, paths) and add the architecture
    # switches. A hand-written subset used to drop keys such as SEED and MIN_LESION_MM3, which
    # the dataset and post-processing then failed on at the first real training step.
    BASE_CFG = dict(CFG)
    BASE_CFG.update(dict(
        CLASSES=["bg", "kidney", "tumor", "cyst"], LESION_CLASSES=[2, 3], KIDNEY_CLASS=1,
        WINDOWS=[(40, 400), (30, 150), (150, 600), (400, 1800)], SINGLE_WINDOW=(40, 400),
        CE_WEIGHTS=[0.5, 1.0, 2.0, 2.0],
        N_SLICES=CFG["N_SLICES"], N_QUERIES=CFG["N_QUERIES"], D_MODEL=CFG["D_MODEL"],
        N_DEC_LAYERS=CFG["N_DEC_LAYERS"], MASK_DIM=CFG["MASK_DIM"], KNN=CFG["KNN"],
        FUSION_BETA=CFG["FUSION_BETA"], TF_P0=CFG["TF_P0"],
        W_QUERY=CFG["W_QUERY"], W_HM=CFG["W_HM"], W_TV=CFG["W_TV"],
        HM_WARMUP_EPOCHS=CFG["HM_WARMUP_EPOCHS"],
        ENHANCE="pawe", AIFI=True, QUERY=True, GRAPH="graph", FUSION=True,
        PRETRAINED=True, WIDTH_MULT=1.0,
    ))

    def make_cfg(**over):
        c = dict(BASE_CFG); c.update(over); return c

    # ------------------------------------------------------------------ building blocks
    def sincos_points(p, dim, temp=10000.0):
        """p: (B, N, 2) normalised (x, y) -> (B, N, dim) positional encoding."""
        d = max(dim // 4, 1)
        omega = 1.0 / temp ** (torch.arange(d, device=p.device, dtype=torch.float32) / d)
        ox = p[..., 0:1] * 2 * math.pi * 10 * omega
        oy = p[..., 1:2] * 2 * math.pi * 10 * omega
        return F.pad(torch.cat([ox.sin(), ox.cos(), oy.sin(), oy.cos()], -1), (0, dim - 4 * d))

    def sincos_2d(h, w, dim, device):
        y, x = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32),
                              torch.arange(w, device=device, dtype=torch.float32), indexing="ij")
        return sincos_points(torch.stack([(x.flatten() + 0.5) / w, (y.flatten() + 0.5) / h], -1)[None], dim)

    class PAWE(nn.Module):
        """Multi-window HU enhancement: soft windows -> LE curve -> noise-gated residual.

        The learned parameters are the window level/width of each of K windows plus a per-pixel
        curve. Calling this "physics-aware" is only defensible if you state exactly this; the
        reviewer's terminology note (report Sec.2) asks for "HU-adaptive multi-window enhancement".
        """

        def __init__(self, windows, n_slices, mode="pawe", n_iter=4, hidden=24):
            super().__init__()
            self.mode, self.S, self.n_iter, self.K = mode, n_slices, n_iter, len(windows)
            L = torch.tensor([w[0] for w in windows], dtype=torch.float32)
            Wd = torch.tensor([w[1] for w in windows], dtype=torch.float32)
            if mode in ("pawe", "pawe_nogate", "wso"):
                self.level = nn.Parameter(L); self.log_width = nn.Parameter(Wd.log())
            else:
                self.register_buffer("level", L); self.register_buffer("log_width", Wd.log())
            if mode in ("pawe", "pawe_nogate"):
                C = self.K * n_slices
                self.curve = nn.Sequential(nn.Conv2d(C, hidden, 3, padding=1), nn.ReLU(True),
                                           nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(True),
                                           nn.Conv2d(hidden, self.K * n_iter, 3, padding=1), nn.Tanh())
                nn.init.zeros_(self.curve[-2].weight); nn.init.zeros_(self.curve[-2].bias)
                self.dn = nn.Conv2d(C, C, 3, padding=1, groups=C)
                nn.init.zeros_(self.dn.weight); nn.init.zeros_(self.dn.bias)
                self.gate = nn.Conv2d(C, C, 1)
            self.out_ch = self.K * n_slices

        def forward(self, hu):
            with torch.autocast(device_type=hu.device.type, enabled=False):
                hu = hu.float()
                B, S, H, W = hu.shape
                L = self.level.view(1, -1, 1, 1, 1)
                Wd = self.log_width.clamp(math.log(20.0), math.log(4000.0)).exp().view(1, -1, 1, 1, 1)
                y = torch.sigmoid(8.0 * (hu[:, None] - L) / Wd)
                alpha = None
                if self.mode in ("pawe", "pawe_nogate"):
                    a = self.curve(y.reshape(B, -1, H, W)).view(B, self.n_iter, self.K, 1, H, W)
                    for n in range(self.n_iter):
                        y = y + a[:, n] * y * (1 - y)
                    f = y.reshape(B, -1, H, W)
                    if self.mode == "pawe":
                        mu = F.avg_pool2d(f, 5, 1, 2)
                        sd = (F.avg_pool2d(f * f, 5, 1, 2) - mu * mu).clamp_min(1e-6).sqrt()
                        f = f + torch.sigmoid(self.gate(sd * 10.0)) * self.dn(f)
                    alpha = a
                else:
                    f = y.reshape(B, -1, H, W)
                return (f - 0.5) / 0.25, alpha, y

    class AIFI(nn.Module):
        """Attention-based intra-scale feature interaction on the top 2-D feature map."""

        def __init__(self, c_in, d, nhead=8):
            super().__init__()
            self.proj = nn.Conv2d(c_in, d, 1)
            self.layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout=0.0,
                                                    batch_first=True, norm_first=True)

        def forward(self, f):
            x = self.proj(f); B, C, H, W = x.shape
            t = x.flatten(2).transpose(1, 2)
            t = self.layer(t + sincos_2d(H, W, C, x.device))
            return t.transpose(1, 2).reshape(B, C, H, W)

    def cbr(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))

    class Up(nn.Module):
        def __init__(self, i, s, o):
            super().__init__(); self.c = nn.Sequential(cbr(i + s, o), cbr(o, o))

        def forward(self, x, skip=None):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            if skip is not None:
                if skip.shape[-2:] != x.shape[-2:]:
                    x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
                x = torch.cat([x, skip], 1)
            return self.c(x)

    class SAGE(nn.Module):
        """Mean-aggregator GraphSAGE layer on a dense, row-normalised adjacency."""

        def __init__(self, d):
            super().__init__()
            self.norm = nn.LayerNorm(2 * d)
            self.mlp = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))

        def forward(self, h, A):
            return h + self.mlp(self.norm(torch.cat([h, torch.bmm(A, h)], -1)))

    class DecLayer(nn.Module):
        def __init__(self, d, graph_mode, nhead=8):
            super().__init__()
            self.graph_mode = graph_mode
            if graph_mode == "graph":
                self.sage = SAGE(d)
            elif graph_mode == "selfattn":
                self.n_sa = nn.LayerNorm(d); self.sa = nn.MultiheadAttention(d, nhead, batch_first=True)
            self.n_ca = nn.LayerNorm(d); self.ca = nn.MultiheadAttention(d, nhead, batch_first=True)
            self.n_ff = nn.LayerNorm(d); self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

        def forward(self, nodes, qpos, K, mem, mpos, A):
            if self.graph_mode == "graph":
                nodes = self.sage(nodes, A)
            q = nodes[:, :K]
            if self.graph_mode == "selfattn":
                t = self.n_sa(q); q = q + self.sa(t + qpos, t + qpos, t)[0]
            t = self.n_ca(q); q = q + self.ca(t + qpos, mem + mpos, mem)[0]
            q = q + self.ff(self.n_ff(q))
            return torch.cat([q, nodes[:, K:]], 1)

    def build_graph(qp, knn=6, tau=0.08, grid=4):
        """Three-level graph: K lesion queries + grid*grid region nodes + 1 kidney node."""
        B, K, _ = qp.shape; R = grid * grid; N = K + R + 1; dev = qp.device
        A = torch.zeros(B, N, N, device=dev)
        d = torch.cdist(qp, qp) + torch.eye(K, device=dev)[None] * 1e3
        kk = min(knn, K - 1)
        if kk > 0:
            dv, di = d.topk(kk, dim=-1, largest=False)
            A[:, :K, :K].scatter_(2, di, torch.exp(-dv ** 2 / (2 * tau ** 2)))
            A[:, :K, :K] = torch.maximum(A[:, :K, :K], A[:, :K, :K].transpose(1, 2))
        g = (torch.arange(grid, device=dev, dtype=torch.float32) + 0.5) / grid
        gy, gx = torch.meshgrid(g, g, indexing="ij")
        rc = torch.stack([gx.flatten(), gy.flatten()], -1)
        _, ri = torch.cdist(qp, rc[None].expand(B, -1, -1)).topk(2, dim=-1, largest=False)
        qr = torch.zeros(B, K, R, device=dev).scatter_(2, ri, 1.0)
        A[:, :K, K:K + R] = qr; A[:, K:K + R, :K] = qr.transpose(1, 2)
        A[:, K:K + R, K:K + R] = (torch.cdist(rc, rc) < (1.0 / grid) * 1.01).float() - torch.eye(R, device=dev)
        A[:, -1, :-1] = 1.0; A[:, :-1, -1] = 1.0
        return A / A.sum(-1, keepdim=True).clamp_min(1e-6)

    class MWHQNet(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg
            self.C = len(cfg["CLASSES"]); self.L = len(cfg["LESION_CLASSES"])
            wm = cfg.get("WIDTH_MULT", 1.0)
            wins = [cfg["SINGLE_WINDOW"]] if cfg["ENHANCE"] == "single" else cfg["WINDOWS"]
            mode = {"pawe": "pawe", "pawe_nogate": "pawe_nogate", "wso": "wso"}.get(cfg["ENHANCE"], "fixed")
            self.pawe = PAWE(wins, cfg["N_SLICES"], mode=mode)
            try:
                self.enc = timm.create_model("resnet34", features_only=True, pretrained=cfg["PRETRAINED"],
                                             in_chans=self.pawe.out_ch)
            except Exception as e:
                print(f"  pretrained weights unavailable, training from scratch: {type(e).__name__}")
                self.enc = timm.create_model("resnet34", features_only=True, pretrained=False,
                                             in_chans=self.pawe.out_ch)
            ch = self.enc.feature_info.channels()
            D = max(int(round(cfg["D_MODEL"] * wm)), 32)
            self.D = D
            self.top = AIFI(ch[4], D) if cfg["AIFI"] else nn.Conv2d(ch[4], D, 1)
            self.u16 = Up(D, ch[3], int(128 * wm)); self.u8 = Up(int(128 * wm), ch[2], int(96 * wm))
            self.u4 = Up(int(96 * wm), ch[1], int(64 * wm)); self.u2 = Up(int(64 * wm), ch[0], int(48 * wm))
            self.u1 = Up(int(48 * wm), 0, int(32 * wm))
            self.dense_head = nn.Conv2d(int(32 * wm), self.C, 1)
            if cfg["QUERY"]:
                q_in_dim = int(48 * wm) + int(96 * wm) + self.pawe.K
                self.hm_head = nn.Sequential(cbr(int(48 * wm), int(48 * wm)), nn.Conv2d(int(48 * wm), 1, 1))
                nn.init.constant_(self.hm_head[-1].bias, -2.19)
                self.pix = nn.Sequential(cbr(int(48 * wm), cfg["MASK_DIM"]),
                                         nn.Conv2d(cfg["MASK_DIM"], cfg["MASK_DIM"], 1))
                self.mem16 = nn.Conv2d(int(128 * wm), D, 1); self.mem8 = nn.Conv2d(int(96 * wm), D, 1)
                self.lvl = nn.Parameter(torch.zeros(2, D))
                self.q_in = nn.Sequential(nn.Linear(q_in_dim, D), nn.GELU(), nn.Linear(D, D))
                self.q_pos = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))
                self.region_proj = nn.Linear(D, D); self.kidney_proj = nn.Linear(D, D)
                self.layers = nn.ModuleList([DecLayer(D, cfg["GRAPH"]) for _ in range(cfg["N_DEC_LAYERS"])])
                self.out_norm = nn.LayerNorm(D)
                self.cls_head = nn.Linear(D, self.L + 1)
                self.mask_head = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, cfg["MASK_DIM"]))

        def _heads(self, q, pix):
            h = self.out_norm(q)
            return self.cls_head(h), torch.einsum("bkd,bdhw->bkhw", self.mask_head(h), pix)

        def forward(self, x, gt_cidx=None, tf_p=0.0):
            B, S, H, W = x.shape
            f, alpha, ywin = self.pawe(x)
            e = self.enc(f)
            D = self.D
            p32 = self.top(e[4]); p16 = self.u16(p32, e[3]); p8 = self.u8(p16, e[2])
            p4 = self.u4(p8, e[1]); p2 = self.u2(p4, e[0]); p1 = self.u1(p2)
            if p1.shape[-2:] != (H, W):
                p1 = F.interpolate(p1, size=(H, W), mode="bilinear", align_corners=False)
            out = dict(dense=self.dense_head(p1), alpha=alpha)
            if not self.cfg["QUERY"]:
                return out
            K = self.cfg["N_QUERIES"]
            hm_logit = self.hm_head(p2); out["hm"] = hm_logit
            h2, w2 = p2.shape[-2:]
            with torch.no_grad():
                hm = hm_logit.float().sigmoid()
                peaks = hm * (F.max_pool2d(hm, 3, 1, 1) == hm).float()
                idx = peaks.flatten(1).topk(K, dim=1).indices
                if self.training and gt_cidx is not None and tf_p > 0:
                    idx = idx.clone()
                    for b in range(B):
                        g = gt_cidx[b]
                        if len(g) and _random.random() < tf_p:
                            g = g[: K // 2].to(idx.device)
                            keep = idx[b][~torch.isin(idx[b], g)][: K - len(g)]
                            idx[b] = torch.cat([keep, g])[:K]
            ys, xs = (idx // w2).float(), (idx % w2).float()
            qp = torch.stack([(xs + 0.5) / w2, (ys + 0.5) / h2], -1)
            f2 = torch.gather(p2.flatten(2), 2, idx[:, None].expand(-1, p2.shape[1], -1)).transpose(1, 2)
            f8 = F.grid_sample(p8, (qp * 2 - 1)[:, :, None], align_corners=False)[..., 0].transpose(1, 2)
            yc = ywin[:, :, S // 2].float()
            yc = F.adaptive_avg_pool2d(F.avg_pool2d(yc, 5, 1, 2), (h2, w2))
            fw = torch.gather(yc.flatten(2), 2, idx[:, None].expand(-1, yc.shape[1], -1)).transpose(1, 2)
            q = self.q_in(torch.cat([f2, f8, fw.to(f2.dtype)], -1))
            qpos = self.q_pos(sincos_points(qp, q.shape[-1]).to(q.dtype))
            m16 = self.mem16(p16); m8 = self.mem8(p8)
            mem = torch.cat([m16.flatten(2).transpose(1, 2), m8.flatten(2).transpose(1, 2)], 1)
            mpos = torch.cat([sincos_2d(*m16.shape[-2:], m16.shape[1], x.device) + self.lvl[0],
                              sincos_2d(*m8.shape[-2:], m8.shape[1], x.device) + self.lvl[1]], 1).to(mem.dtype)
            region = self.region_proj(F.adaptive_avg_pool2d(m16, 4).flatten(2).transpose(1, 2))
            with torch.no_grad():
                fgp = 1 - out["dense"].float().softmax(1)[:, :1]
                wk = F.adaptive_avg_pool2d(fgp, m16.shape[-2:]).flatten(1)
                wk = (wk / wk.sum(1, keepdim=True).clamp_min(1e-6)).to(m16.dtype)
            kidney = self.kidney_proj(torch.bmm(m16.flatten(2), wk[..., None]).transpose(1, 2))
            nodes = torch.cat([q, region, kidney], 1)
            A = build_graph(qp.float(), self.cfg["KNN"]).to(nodes.dtype) if self.cfg["GRAPH"] == "graph" else None
            pix = self.pix(p2)
            aux = []
            for layer in self.layers:
                nodes = layer(nodes, qpos, K, mem, mpos, A)
                aux.append(self._heads(nodes[:, :K], pix))
            out.update(cls=aux[-1][0], masks=aux[-1][1], aux=aux, qp=qp)
            return out

    def fuse(out, cfg, renormalize=True):
        """Dense softmax with lesion channels blended towards the query masks.

        FIX (action-plan Sec.4.2): after blending, the categorical distribution no longer sums to 1.
        Renormalising makes probability thresholds and calibration meaningful again; it cannot
        change an argmax, which is why the legacy results happened to survive without it.
        """
        pd_ = out["dense"].float().softmax(1)
        if not (cfg["QUERY"] and cfg["FUSION"]):
            return pd_
        H, W = pd_.shape[-2:]
        pc = out["cls"].float().softmax(-1)[..., :-1]
        pm = F.interpolate(out["masks"].float(), size=(H, W), mode="bilinear", align_corners=False).sigmoid()
        pd_ = pd_.clone(); b = cfg["FUSION_BETA"]
        for j, c in enumerate(cfg["LESION_CLASSES"]):
            pq = (pc[:, :, j, None, None] * pm).amax(1)
            pd_[:, c] = (1 - b) * pd_[:, c] + b * pq
        if renormalize:
            pd_ = pd_.clamp_min(1e-8)
            pd_ = pd_ / pd_.sum(1, keepdim=True).clamp_min(1e-8)
        return pd_

    # ------------------------------------------------------------------ efficiency
    def count_params(model):
        return sum(p.numel() for p in model.parameters())

    @torch.no_grad()
    def measure_cost(model, cfg, size=(5, 192, 192), reps=5):
        """Parameters, MACs and peak VRAM for one 2.5-D slice stack."""
        model.eval()
        x = torch.zeros(1, *size, device=DEV)
        macs = [0]

        def hook(m, i, o):
            if isinstance(m, nn.Conv2d):
                macs[0] += o.shape[-1] * o.shape[-2] * m.out_channels * (m.in_channels // m.groups) \
                           * m.kernel_size[0] * m.kernel_size[1]
            elif isinstance(m, nn.Linear):
                macs[0] += m.in_features * m.out_features

        hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
        if DEV == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model(x)
        for h in hs:
            h.remove()
        t0 = time.time()
        for _ in range(reps):
            model(x)
        if DEV == "cuda":
            torch.cuda.synchronize()
        ms = (time.time() - t0) / reps * 1000
        vram = (torch.cuda.max_memory_allocated() / 1e9) if DEV == "cuda" else 0.0
        return dict(params_M=round(count_params(model) / 1e6, 2),
                    macs_G=round(macs[0] / 1e9, 2), slice_ms=round(ms, 1), peak_vram_GB=round(vram, 2))

    # ------------------------------------------------------------------ tests
    def _model_tests():
        torch.manual_seed(0)
        cfg = make_cfg(PRETRAINED=False)
        m = MWHQNet(cfg).to(DEV).eval()
        x = torch.zeros(2, cfg["N_SLICES"], 192, 192, device=DEV)
        with torch.no_grad():
            out = m(x)
        assert out["dense"].shape == (2, 4, 192, 192), out["dense"].shape
        p = fuse(out, cfg)
        s = p.sum(1)
        assert torch.allclose(s, torch.ones_like(s), atol=1e-4), f"fusion does not sum to 1: {s.min()}, {s.max()}"
        p_raw = fuse(out, cfg, renormalize=False)
        assert not torch.allclose(p_raw.sum(1), torch.ones_like(s), atol=1e-4), "renormalisation had no effect"
        A = build_graph(torch.rand(2, 32, 2, device=DEV), knn=6)
        assert A.shape == (2, 32 + 16 + 1, 32 + 16 + 1)
        assert torch.allclose(A.sum(-1), torch.ones_like(A.sum(-1)), atol=1e-5), "adjacency is not row-normalised"
        # argmax must be unchanged by renormalisation - the reason the legacy results survived
        assert torch.equal(p.argmax(1), p_raw.argmax(1)), "renormalisation changed the argmax"
        return dict(dense_out=tuple(out["dense"].shape), queries=tuple(out["masks"].shape), adjacency=tuple(A.shape))

    _t = _model_tests()
    print(f"model tests passed: dense {_t['dense_out']}, query masks {_t['queries']}, graph {_t['adjacency']}")

    # ------------------------------------------------------------------ registry
    EXPERIMENTS = {
        # id: (description, cfg overrides)
        "B0_baseline":     ("2.5-D ResNet-34 U-Net, single window, no query/graph (your legacy baseline)",
                            dict(ENHANCE="single", AIFI=False, QUERY=False, FUSION=False)),
        "B0_wide":         ("capacity-matched baseline (~B1 parameter count) - reviewer Sec.1.5",
                            dict(ENHANCE="single", AIFI=False, QUERY=False, FUSION=False, WIDTH_MULT=1.35)),
        "B1_full":         ("PAWE + LCQS + HQGD, fixed fusion beta (the proposed model)",
                            dict(ENHANCE="pawe", AIFI=True, QUERY=True, GRAPH="graph", FUSION=True)),
        "A1_fixed_multi":  ("B0 + four fixed HU windows, no learned curves (multi-window control)",
                            dict(ENHANCE="fixed_multi", AIFI=True, QUERY=True, GRAPH="none", FUSION=True)),
        "A2_wso":          ("B0 + learned window level/width only, no curves or gating",
                            dict(ENHANCE="wso", AIFI=True, QUERY=True, GRAPH="none", FUSION=True)),
        "A2b_pawe_nogate": ("PAWE without the noise-gated residual (isolates the gate)",
                            dict(ENHANCE="pawe_nogate", AIFI=True, QUERY=True, GRAPH="none", FUSION=True)),
        "A3_no_graph":     ("PAWE + LCQS with GRAPH='none' (queries, no graph, no self-attention)",
                            dict(ENHANCE="pawe", AIFI=True, QUERY=True, GRAPH="none", FUSION=True)),
        "A4_selfattn":     ("PAWE + LCQS + self-attention decoder - the graph-vs-attention test",
                            dict(ENHANCE="pawe", AIFI=True, QUERY=True, GRAPH="selfattn", FUSION=True)),
        "A5_full_graph":   ("same as B1, listed so the ablation table names the full model explicitly",
                            dict(ENHANCE="pawe", AIFI=True, QUERY=True, GRAPH="graph", FUSION=True)),
        "A6_no_fusion":    ("B1 with the query masks switched out of the fusion",
                            dict(ENHANCE="pawe", AIFI=True, QUERY=True, GRAPH="graph", FUSION=False)),
        "A7_no_aifi":      ("B1 without the AIFI transformer block",
                            dict(ENHANCE="pawe", AIFI=False, QUERY=True, GRAPH="graph", FUSION=True)),
        "A8_single_win":   ("full architecture fed a single HU window (does PAWE matter at all?)",
                            dict(ENHANCE="single", AIFI=True, QUERY=True, GRAPH="graph", FUSION=True)),
    }
    print(f"\n{len(EXPERIMENTS)} experiment configurations registered:")

    for k, (desc, ov) in EXPERIMENTS.items():
        mm = MWHQNet(make_cfg(PRETRAINED=False, **ov))
        print(f"  {k:18s} {count_params(mm) / 1e6:6.2f} M   {desc}")
        del mm
    if DEV == "cuda":
        torch.cuda.empty_cache()
