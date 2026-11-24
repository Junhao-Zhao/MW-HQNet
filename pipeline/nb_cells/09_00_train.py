# ============================================================================
# CELL 9 -Dataset and the training loop with INNER-DEV checkpoint selection
#
# The one behavioural change that matters for reviewer 1.6: the loop evaluates on
# `dev` (inner development), never on `outer`. The outer fold is not touched here at all;
# it is scored once in Cell 11/12 after the checkpoint is frozen.
# ============================================================================
if ENABLED["train"] and MODEL_NEEDED:
    from torch.utils.data import Dataset, DataLoader
    import cv2

    def build_spacing_map(df):
        return {r["key"]: tuple(float(x) for x in r["spacing"]) for _, r in df.iterrows()}

    def lowdose_noise(hu, dose, rng, sigma0=12.0):
        if dose >= 0.999:
            return hu
        s = sigma0 * math.sqrt(1.0 / dose - 1.0)
        n = ndi.gaussian_filter(rng.standard_normal(hu.shape).astype(np.float32), sigma=(0, 0.8, 0.8))
        n /= n.std() + 1e-6
        return hu + s * n

    def lesion_instances(sem, lesion_classes, max_inst=24):
        masks, labels, centres = [], [], []
        for li, c in enumerate(lesion_classes):
            cc, n = ndi.label(sem == c)
            for i in range(1, n + 1):
                m = cc == i
                if m.sum() < 1:
                    continue
                dt = ndi.distance_transform_edt(m)
                cy, cx = np.unravel_index(np.argmax(dt), dt.shape)
                masks.append(m.astype(np.uint8)); labels.append(li)
                centres.append((cy, cx, math.sqrt(m.sum() / math.pi)))
        if len(masks) > max_inst:
            order = np.argsort([-m.sum() for m in masks])[:max_inst]
            masks = [masks[i] for i in order]; labels = [labels[i] for i in order]
            centres = [centres[i] for i in order]
        return masks, labels, centres

    def centre_heatmap(centres, H, W, stride=2):
        h, w = H // stride, W // stride
        hm = np.zeros((h, w), np.float32)
        yy, xx = np.mgrid[:h, :w]
        for cy, cx, r in centres:
            py, px = min(int(cy // stride), h - 1), min(int(cx // stride), w - 1)
            sig = max(0.5 * r / stride, 1.0)
            g = np.exp(-((yy - py) ** 2 + (xx - px) ** 2) / (2 * sig ** 2))
            hm = np.maximum(hm, g); hm[py, px] = 1.0
        return hm

    class SliceDS(Dataset):
        """2.5-D slices sampled from ROI volumes; lesion-centric oversampling kept as in the legacy code."""

        def __init__(self, rows, cfg, train):
            self.cfg, self.train, self.S = cfg, train, cfg["N_SLICES"]
            self.keys = list(rows["key"])
            self.img = dict(zip(rows["key"], rows["img"]))
            self.seg = dict(zip(rows["key"], rows["seg"]))
            self._cache = {}
            self.index, self.lesion_index = [], []
            les = np.array(cfg["LESION_CLASSES"])
            for v, k in enumerate(self.keys):
                sg = self._get(k, "seg")
                for z in range(sg.shape[0]):
                    self.index.append((v, z))
                    if np.isin(sg[z], les).any():
                        self.lesion_index.append((v, z))
            self.rng = np.random.default_rng(cfg["SEED"])

        def _get(self, key, which):
            ck = (key, which)
            if ck not in self._cache:
                p = self.img[key] if which == "img" else self.seg[key]
                self._cache[ck] = np.load(p, mmap_mode="r")
            return self._cache[ck]

        def __len__(self):
            return self.cfg["ITERS_PER_EPOCH"] * self.cfg["BATCH"] if self.train else len(self.index)

        def _stack(self, v, z):
            k = self.keys[v]
            im, sg = self._get(k, "img"), self._get(k, "seg")
            idx = np.clip(np.arange(z - self.S // 2, z + self.S // 2 + 1), 0, im.shape[0] - 1)
            return np.asarray(im[idx], np.float32), np.asarray(sg[z], np.int64), sg.shape

        def _augment(self, x, y):
            r = self.rng
            S, H, W = x.shape
            if r.random() < 0.5:
                x, y = x[:, :, ::-1], y[:, ::-1]
            if r.random() < 0.7:
                ang, sc = r.uniform(-15, 15), r.uniform(0.85, 1.15)
                M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, sc); M[:, 2] += r.uniform(-8, 8, 2)
                x = np.stack([cv2.warpAffine(np.ascontiguousarray(c), M, (W, H), flags=cv2.INTER_LINEAR,
                                             borderValue=float(x.min())) for c in x])
                y = cv2.warpAffine(np.ascontiguousarray(y).astype(np.uint8), M, (W, H),
                                   flags=cv2.INTER_NEAREST, borderValue=0).astype(np.int64)
            if r.random() < 0.5:
                x = x * r.uniform(0.92, 1.08) + r.uniform(-20.0, 20.0)
            if r.random() < 0.3:
                x = lowdose_noise(x, r.uniform(0.1, 0.5), r, 12.0)
            if r.random() < 0.15:
                x = ndi.gaussian_filter(x, sigma=(0, r.uniform(0.5, 1.0), r.uniform(0.5, 1.0)))
            return np.ascontiguousarray(x, np.float32), np.ascontiguousarray(y)

        def __getitem__(self, i):
            if self.train:
                pool = self.lesion_index if (self.lesion_index and self.rng.random() < 0.6) else self.index
                v, z = pool[self.rng.integers(len(pool))]
            else:
                v, z = self.index[i]
            x, y, _ = self._stack(v, z)
            if self.train:
                x, y = self._augment(x, y)
            masks, labels, centres = lesion_instances(y, self.cfg["LESION_CLASSES"])
            H, W = y.shape
            hm = centre_heatmap(centres, H, W)
            cidx = [min(int(c[0] // 2), H // 2 - 1) * (W // 2) + min(int(c[1] // 2), W // 2 - 1) for c in centres]
            return dict(x=torch.from_numpy(x), y=torch.from_numpy(y), hm=torch.from_numpy(hm)[None],
                        inst_masks=torch.from_numpy(np.stack(masks)) if masks else torch.zeros(0, H, W, dtype=torch.uint8),
                        inst_labels=torch.tensor(labels, dtype=torch.long),
                        inst_cidx=torch.tensor(cidx, dtype=torch.long))

    def collate(batch):
        out = {k: torch.stack([b[k] for b in batch]) for k in ("x", "y", "hm")}
        for k in ("inst_masks", "inst_labels", "inst_cidx"):
            out[k] = [b[k] for b in batch]
        return out

    def evaluate(model, cfg, rows, tta=False, limit=None):
        case_rows, les_rows = [], []
        rows = rows if limit is None else rows.head(limit)
        for _, r in rows.iterrows():
            img = np.load(r["img"])
            gt = np.load(r["seg"])
            probs = predict_volume(model, img, cfg, tta=tta)
            lab = postprocess(probs, r["spacing"], cfg)
            c, l = score_roi(lab, gt, r["spacing"], cfg, r["key"])
            case_rows.append(c); les_rows.extend(l)
        return pd.DataFrame(case_rows), pd.DataFrame(les_rows)

    # ------------------------------------------------------------- training
    SEL_METRIC = "Dice masses"

    def train_run(exp_id, fold, max_epochs=None):
        """Train one experiment on one outer fold, selecting the checkpoint on inner-dev only."""
        from tqdm.auto import tqdm
        desc, ov = EXPERIMENTS[exp_id]
        cfg = make_cfg(**ov); cfg["AMP"] = CFG["AMP"]
        run_dir = DIRS["runs"] / f"{exp_id}_fold{fold}"
        run_dir.mkdir(parents=True, exist_ok=True)
        json.dump({k: v for k, v in cfg.items()}, open(run_dir / "config.json", "w"), indent=1, default=str)
        json.dump(dict(experiment=exp_id, fold=fold, description=desc,
                       fit=len(SPLITS[fold]["fit"]), dev=len(SPLITS[fold]["dev"]),
                       outer=len(SPLITS[fold]["outer"]), track=CFG["TRACK"]),
                  open(run_dir / "split.json", "w"), indent=1)

        if (run_dir / "DONE").exists():
            print(f"  {exp_id} fold{fold}: already finished, skipping")
            return run_dir

        dev_rows, fit_rows = FOLD_ROIS[fold]["dev"], FOLD_ROIS[fold]["fit"]
        if len(fit_rows) == 0 or len(dev_rows) == 0:
            print(f"  {exp_id} fold{fold}: no ROIs available - run Cell 4/5 first")
            return None

        torch.manual_seed(CFG["SEED"] + fold)
        model = MWHQNet(cfg).to(DEV)
        print(f"  {exp_id} fold{fold}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M params, "
              f"fit {len(fit_rows)} ROIs, dev {len(dev_rows)} ROIs")

        ds = SliceDS(fit_rows, cfg, train=True)
        dl = DataLoader(ds, batch_size=cfg["BATCH"], shuffle=False, num_workers=CFG["NUM_WORKERS"],
                        collate_fn=collate, pin_memory=True, drop_last=True,
                        persistent_workers=CFG["NUM_WORKERS"] > 0)
        pawe_p = [p for n, p in model.named_parameters() if n.startswith("pawe.")]
        other = [p for n, p in model.named_parameters() if not n.startswith("pawe.")]
        opt = torch.optim.AdamW([{"params": other}, {"params": pawe_p, "lr": CFG["LR"] * 3, "weight_decay": 0.0}],
                                lr=CFG["LR"], weight_decay=CFG["WD"])
        epochs = max_epochs or CFG["EPOCHS"]
        total_steps = epochs * CFG["ITERS_PER_EPOCH"]
        warm = CFG["WARMUP_EPOCHS"] * CFG["ITERS_PER_EPOCH"]
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / max(warm, 1)) * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))
        scaler = torch.amp.GradScaler("cuda", enabled=cfg["AMP"] and DEV == "cuda")
        start_ep, best = 0, -1.0
        ck_last, ck_best = run_dir / "last.pt", run_dir / "best.pt"
        if ck_last.exists():
            ck = torch.load(ck_last, map_location=DEV, weights_only=False)
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
            start_ep, best = ck["epoch"] + 1, ck["best"]
            print(f"    resumed at epoch {start_ep}, best dev {best:.4f}")

        log = open(run_dir / "log.jsonl", "a")
        for ep in tqdm(range(start_ep, epochs), desc=f"    {exp_id} f{fold}", leave=False):
            model.train(); t0 = time.time(); run = {}
            tf_p = cfg["TF_P0"] * max(0.0, 1 - ep / (0.5 * epochs))
            ramp = min(1.0, (ep + 1) / max(cfg["HM_WARMUP_EPOCHS"], 1))
            cfg_ep = dict(cfg); cfg_ep["W_HM"] = cfg["W_HM"] * ramp; cfg_ep["W_QUERY"] = cfg["W_QUERY"] * ramp
            for it, batch in enumerate(dl):
                if it >= cfg["ITERS_PER_EPOCH"]:
                    break
                x = batch["x"].to(DEV, non_blocking=True)
                batch["y"] = batch["y"].to(DEV, non_blocking=True)
                batch["hm"] = batch["hm"].to(DEV, non_blocking=True)
                with torch.autocast(device_type=DEV, enabled=cfg["AMP"] and DEV == "cuda"):
                    out = model(x, gt_cidx=batch["inst_cidx"], tf_p=tf_p)
                loss, parts = total_loss(out, batch, cfg_ep)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["GRAD_CLIP"])
                scaler.step(opt); scaler.update(); sched.step()
                for k, v in parts.items():
                    run[k] = run.get(k, 0.0) + float(v)
            rec = dict(epoch=ep, time_min=round((time.time() - t0) / 60, 2),
                       **{k: v / (it + 1) for k, v in run.items()})
            if isinstance(model.pawe.level, torch.nn.Parameter):
                rec["windows"] = [[round(float(l), 1), round(float(w), 1)]
                                  for l, w in zip(model.pawe.level.detach().cpu(),
                                                  model.pawe.log_width.detach().exp().cpu())]
            if (ep + 1) % CFG["VAL_EVERY"] == 0 or ep == epochs - 1:
                dfc, dfl = evaluate(model, cfg, dev_rows, tta=False)
                summ = summarise(dfc.to_dict("records"), dfl.to_dict("records"), cfg)
                rec.update({f"dev_{k}": v for k, v in summ.items()})
                if summ.get(SEL_METRIC, -1) > best:
                    best = summ[SEL_METRIC]
                    torch.save(model.state_dict(), ck_best)
            rec["best_dev"] = best
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                            scaler=scaler.state_dict(), epoch=ep, best=best), ck_last)
            log.write(json.dumps(rec, default=str) + "\n"); log.flush()
        log.close()
        (run_dir / "DONE").write_text("ok")
        print(f"    {exp_id} fold{fold}: best inner-dev {SEL_METRIC} = {best:.4f}")
        return run_dir

    TRAIN_ROWS = {f: FOLD_ROIS[f]["fit"] for f in CFG["FOLDS"]}
    print("training scaffolding ready: train_run(experiment_id, fold) selects on inner-dev only")
    print(f"select metric = {SEL_METRIC!r}   (prespecified; change it only before you run, never after)")
