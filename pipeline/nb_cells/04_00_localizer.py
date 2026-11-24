# ============================================================================
# CELL 4 - Label-independent kidney localiser + coverage audit   [TRACK == "endo"]
#
# Reviewer Sec.1.1 CRITICAL. The fine model cannot see a whole CT, so something must propose
# where the kidneys are. That something must be trained and applied WITHOUT labels, and its
# failures must be measured, not assumed away.
#
# This cell trains a small 2.5-D U-Net on the 2.5 mm cache. Its only input is CT intensity.
# Its output is a kidney probability map, from which we take connected components, keep the
# two largest above 5 mL, pad them by 12 mm, and hand the resulting boxes to Cell 5.
#
# The coverage audit at the end is the evidence the reviewer asks for: kidney recall, box
# coverage of every ground-truth lesion, and the failure rate. Those numbers go in Table II
# of the revised paper as "localisation stage" performance.
# ============================================================================
if ENABLED["localizer"] and CFG["TRACK"] == "endo":
    import zipfile
    torch = ensure_torch()
    import torch.nn as nn
    import torch.nn.functional as F
    from scipy import ndimage as ndi

    DS_MM = 2.5
    LOC = dict(epochs=8, iters=400, batch=16, lr=2e-3, base=24, slices=3,
               thresh=0.5, min_ml=5.0, pad_mm=12.0, seed=CFG["SEED"])

    def _cbr(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))

    class KidneyLocalizer(nn.Module):
        """3-slice 2.5-D encoder-decoder, one logit per pixel: 'this pixel is kidney or lesion'."""

        def __init__(self, base=24, n_in=3):
            super().__init__()
            self.e1 = nn.Sequential(_cbr(n_in, base), _cbr(base, base))
            self.e2 = nn.Sequential(_cbr(base, base * 2), _cbr(base * 2, base * 2))
            self.e3 = nn.Sequential(_cbr(base * 2, base * 4), _cbr(base * 4, base * 4))
            self.e4 = nn.Sequential(_cbr(base * 4, base * 8), _cbr(base * 8, base * 8))
            self.d3 = nn.Sequential(_cbr(base * 8 + base * 4, base * 4), _cbr(base * 4, base * 4))
            self.d2 = nn.Sequential(_cbr(base * 4 + base * 2, base * 2), _cbr(base * 2, base * 2))
            self.d1 = nn.Sequential(_cbr(base * 2 + base, base), _cbr(base, base))
            self.head = nn.Conv2d(base, 1, 1)

        def forward(self, x):
            e1 = self.e1(x)
            e2 = self.e2(F.max_pool2d(e1, 2))
            e3 = self.e3(F.max_pool2d(e2, 2))
            e4 = self.e4(F.max_pool2d(e3, 2))
            d3 = self.d3(torch.cat([F.interpolate(e4, size=e3.shape[-2:], mode="bilinear", align_corners=False), e3], 1))
            d2 = self.d2(torch.cat([F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], 1))
            d1 = self.d1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], 1))
            return self.head(d1)

    class LocDS(torch.utils.data.Dataset):
        """Random 3-slice stacks from the image-only cache; the target is kidney U lesion."""

        def __init__(self, cases, train=True, iters=1, batch=1, seed=0):
            self.cases, self.train, self.rng = list(cases), train, np.random.default_rng(seed)
            self.S = LOC["slices"]
            self.n = iters * batch if train else 1
            self.idx = []
            for c in self.cases:
                z = np.load(DIRS["cache_ct"] / f"{c}.npz")["ct"].shape[0]
                self.idx.append((c, z))

        def __len__(self):
            return self.n if self.train else len(self.cases)

        def _stack(self, ct, z):
            S = self.S
            ids = np.clip(np.arange(z - S // 2, z + S // 2 + 1), 0, ct.shape[0] - 1)
            return ct[ids].astype(np.float32)

        def __getitem__(self, i):
            if self.train:
                c = self.cases[self.rng.integers(len(self.cases))]
                ct = np.load(DIRS["cache_ct"] / f"{c}.npz")["ct"]
                z = int(self.rng.integers(0, ct.shape[0]))
                lab = np.load(DIRS["cache_seg"] / f"{c}_ds.npz")["lab"]
                x = self._stack(ct, z)[:, None]
                if self.rng.random() < 0.5:
                    x, y = x[:, :, :, ::-1], lab[z][:, ::-1]
                else:
                    y = lab[z]
                return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy((y > 0).astype(np.float32))[None]
            c = self.cases[i]
            ct = np.load(DIRS["cache_ct"] / f"{c}.npz")["ct"]
            lab = np.load(DIRS["cache_seg"] / f"{c}_ds.npz")["lab"]
            return torch.from_numpy(self._stack(ct, 0)[:, None]), torch.from_numpy((lab > 0).astype(np.float32))[None], c

    def _bce_dice(logit, target):
        bce = F.binary_cross_entropy_with_logits(logit, target)
        p = logit.sigmoid()
        inter = (p * target).sum((1, 2, 3))
        den = p.sum((1, 2, 3)) + target.sum((1, 2, 3))
        return bce + (1 - (2 * inter + 1) / (den + 1)).mean()

    @torch.no_grad()
    def predict_kidney_prob(model, ct, bs=24):
        """Full-volume kidney probability at the 2.5 mm cache grid. Image only, by construction."""
        model.eval()
        Z = ct.shape[0]
        out = np.zeros(ct.shape, np.float32)
        S = LOC["slices"]
        for z0 in range(0, Z, bs):
            zs = np.arange(z0, min(Z, z0 + bs))
            xb = np.stack([ct[np.clip(np.arange(z - S // 2, z + S // 2 + 1), 0, Z - 1)] for z in zs])
            xt = torch.from_numpy(xb[:, :, None]).to(DEV).float()
            with torch.autocast(device_type="cuda", enabled=CFG["AMP"] and DEV == "cuda"):
                logit = model(xt)
            out[z0:z0 + len(zs)] = logit.float().sigmoid()[:, 0].cpu().numpy()
        return out

    def boxes_from_prob(prob, sp_ds):
        """Kidney components -> padded bounding boxes in ORIGINAL-cache index space (image only)."""
        vox = float(np.prod(sp_ds))
        m = prob >= LOC["thresh"]
        cc, n = ndi.label(m)
        if n == 0:
            return []
        sizes = ndi.sum(m, cc, range(1, n + 1)) * vox / 1000.0        # mL
        order = [i + 1 for i in np.argsort(-sizes) if sizes[i] >= LOC["min_ml"]][:2]
        pad = [int(round(LOC["pad_mm"] / s)) for s in sp_ds]
        out = []
        for lid in order:
            idx = np.argwhere(cc == lid)
            lo, hi = idx.min(0), idx.max(0)
            lo = np.maximum(lo - np.array(pad), 0)
            hi = np.minimum(hi + np.array(pad), np.array(prob.shape) - 1)
            out.append(dict(lo=lo.tolist(), hi=hi.tolist(),
                            volume_ml=round(float(sizes[lid - 1]), 1)))
        return out

    def train_localizer(fold, fit_cases, dev_cases, out_dir):
        t = torch
        model = KidneyLocalizer(LOC["base"]).to(DEV)
        opt = t.optim.AdamW(model.parameters(), lr=LOC["lr"], weight_decay=1e-4)
        steps = LOC["epochs"] * LOC["iters"]
        sched = t.optim.lr_scheduler.OneCycleLR(opt, max_lr=LOC["lr"], total_steps=steps, pct_start=0.2)
        scaler = t.amp.GradScaler("cuda", enabled=CFG["AMP"] and DEV == "cuda")
        ds = LocDS(fit_cases, train=True, iters=LOC["iters"], batch=LOC["batch"], seed=CFG["SEED"] + fold)
        dl = t.utils.data.DataLoader(ds, batch_size=LOC["batch"], num_workers=0, drop_last=True)
        best, best_state = -1.0, None

        def dev_dice(eps=0.35):
            """Fast proxy: per-case voxel Dice on the kidney U lesion mask for a few dev cases."""
            scores = []
            for c in dev_cases:
                ct = np.load(DIRS["cache_ct"] / f"{c}.npz")["ct"]
                gt = np.load(DIRS["cache_seg"] / f"{c}_ds.npz")["lab"] > 0
                if gt.sum() < 200:
                    continue
                pr = predict_kidney_prob(model, ct) >= LOC["thresh"]
                scores.append(2 * (pr & gt).sum() / max(pr.sum() + gt.sum(), 1))
                if len(scores) >= 6:
                    break
            return float(np.mean(scores)) if scores else 0.0

        step = 0
        for ep in range(LOC["epochs"]):
            model.train(); run = 0.0
            for x, y in dl:
                x, y = x.to(DEV).float(), y.to(DEV).float()
                with t.autocast(device_type="cuda", enabled=CFG["AMP"] and DEV == "cuda"):
                    loss = _bce_dice(model(x), y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt); t.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sched.step()
                run += float(loss); step += 1
            d = dev_dice()
            print(f"    localizer ep{ep}: loss {run / max(step, 1):.4f}  dev dice {d:.4f}")
            if d > best:
                best = d
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if best_state is not None:
            model.load_state_dict(best_state)
        t.save(dict(state=model.state_dict(), cfg=LOC, fold=fold, dev_dice=best),
               out_dir / "best.pt")
        return model, best

    # ---------------------------------------------------------------- per fold
    folds_json = json.loads(zipfile.ZipFile(CFG["LEGACY_BASELINE_ZIP"]).read(CFG["FOLDS_JSON_ZIP_ENTRY"]))
    FOLD_CASES = {int(k) if k.isdigit() else i: v for i, v in enumerate(folds_json["folds"])}
    cached = {p.stem for p in DIRS["cache_ct"].glob("*.npz")}
    print(f"{len(cached)} cases in the image cache")

    cov_rows = []
    for fold in CFG["FOLDS"]:
        out_dir = DIRS["localizer"] / f"fold{fold}"
        out_dir.mkdir(parents=True, exist_ok=True)
        outer_test = [c for c in FOLD_CASES[fold] if c in cached]
        outer_train = [c for c in sorted(cached) if c not in set(FOLD_CASES[fold])]
        rng = np.random.default_rng(4200 + fold)
        perm = rng.permutation(len(outer_train))
        n_dev = max(1, int(len(outer_train) * CFG["DEV_FRACTION"]))
        dev_cases = [outer_train[i] for i in perm[:n_dev]]
        fit_cases = [outer_train[i] for i in perm[n_dev:]]
        assert not (set(fit_cases) & set(dev_cases)) and not (set(fit_cases) & set(outer_test))

        ck = out_dir / "best.pt"
        if ck.exists():
            st = torch.load(ck, map_location=DEV, weights_only=False)
            model = KidneyLocalizer(LOC["base"]).to(DEV); model.load_state_dict(st["state"])
            print(f"  fold {fold}: reusing localizer (dev dice {st['dev_dice']:.4f})")
        else:
            print(f"  fold {fold}: training localizer on {len(fit_cases)} cases, dev {len(dev_cases)}, "
                  f"outer test {len(outer_test)} untouched")
            model, best = train_localizer(fold, fit_cases, dev_cases, out_dir)

        # ---- coverage audit on the untouched outer fold -------------------------
        model.eval()
        for c in outer_test:
            z = np.load(DIRS["cache_ct"] / f"{c}.npz")
            ct, sp_ds = z["ct"], z["sp"]
            seg = np.load(DIRS["cache_seg"] / f"{c}.npz")
            lab, sp_nat = seg["lab"], seg["sp"]
            prob = predict_kidney_prob(model, ct)
            boxes = boxes_from_prob(prob, sp_ds)
            # map ds boxes to native index space (no label involved)
            ratio = np.asarray(sp_ds, np.float32) / np.asarray(sp_nat, np.float32)
            nat_boxes = [dict(lo=(np.array(b["lo"]) * ratio).astype(int).tolist(),
                              hi=(np.array(b["hi"]) * ratio).astype(int).tolist(),
                              volume_ml=b["volume_ml"]) for b in boxes]
            inside = np.zeros(lab.shape, bool)
            for b in nat_boxes:
                sl = tuple(slice(max(lo, 0), min(hi + 1, n)) for lo, hi, n in zip(b["lo"], b["hi"], lab.shape))
                inside[sl] = True
            gk = lab == 1
            gl = np.isin(lab, [2, 3])
            rec = dict(case=c, fold=fold, n_boxes=len(boxes),
                       box_volume_ml=round(sum(b["volume_ml"] for b in boxes), 1),
                       kidney_recall=round(float((inside & gk).sum()) / max(int(gk.sum()), 1), 4),
                       lesion_recall=(round(float((inside & gl).sum()) / max(int(gl.sum()), 1), 4)
                                      if gl.any() else np.nan),
                       gt_kidney_ml=round(float(gk.sum() * float(np.prod(sp_nat))) / 1000.0, 1),
                       fallback_full=bool(len(boxes) == 0))
            cov_rows.append(rec)
        print(f"  fold {fold}: localisation audited on {len(outer_test)} held-out cases")

    cov = pd.DataFrame(cov_rows)
    cov.to_csv(DIRS["audit"] / "localizer_coverage.csv", index=False)
    print("\nLOCALISER COVERAGE (outer folds, labels used for AUDIT ONLY)")
    print(cov.groupby("fold")[["n_boxes", "kidney_recall", "lesion_recall", "fallback_full"]]
          .mean().round(4).to_string())
    print(f"\noverall kidney recall  : {cov['kidney_recall'].mean():.4f} "
          f"(min {cov['kidney_recall'].min():.4f})")
    print(f"overall lesion recall  : {cov['lesion_recall'].mean():.4f}")
    print(f"cases needing fallback : {int(cov['fallback_full'].sum())} / {len(cov)}")
    print(f"median boxes per case  : {cov['n_boxes'].median():.0f}")
