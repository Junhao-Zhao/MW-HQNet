# ============================================================================
# CELL 8 -Losses, volume inference, post-processing and metric primitives
#
# These are the parts the reviewer's 1.7 and 1.8 target, so they are written to be auditable:
#   * lesion detection is 3-D connected components with a declared, tunable overlap criterion
#     and deterministic one-to-one matching - no more "any overlapping voxel counts as a hit"
#   * surface metrics use the Euclidean distance transform in millimetres, and Normalised
#     Surface Dice is reported next to HD95
#   * the kidney gate is a function of the PREDICTION, never of the label
# ============================================================================
if MODEL_NEEDED:
    from scipy import ndimage as ndi
    from scipy.optimize import linear_sum_assignment

    # ------------------------------------------------------------------ losses
    def dense_loss(logits, y, C, w):
        logits = logits.float()
        ce = F.cross_entropy(logits, y, weight=torch.tensor(w, device=logits.device, dtype=torch.float32))
        p = logits.softmax(1); oh = F.one_hot(y, C).permute(0, 3, 1, 2).float()
        inter = (p * oh).sum((0, 2, 3)); den = p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
        dice = 1 - ((2 * inter + 1) / (den + 1))[1:].mean()
        return ce + dice

    def focal_hm(logit, gt):
        """Centre-heatmap focal loss with separately normalised positive and negative terms."""
        p = logit.float().sigmoid().clamp(1e-4, 1 - 1e-4)
        pos = gt.eq(1).float(); neg = 1 - pos
        lp = -(torch.log(p) * (1 - p) ** 2 * pos).sum() / pos.sum().clamp_min(1)
        ln = -(torch.log(1 - p) * p ** 2 * (1 - gt) ** 4 * neg).sum() / neg.sum().clamp_min(1)
        return lp + ln

    @torch.no_grad()
    def hungarian(cls_logit, mask_logit, gl, gm_lr):
        if len(gl) == 0:
            return np.array([], int), np.array([], int)
        prob = cls_logit.float().softmax(-1)[:, gl]
        x = mask_logit.float().flatten(1); g = gm_lr.flatten(1)
        hw = x.shape[1]
        bce = (F.softplus(-x) @ g.T + F.softplus(x) @ (1 - g).T) / hw
        s = x.sigmoid()
        dice = 1 - (2 * s @ g.T + 1) / (s.sum(1)[:, None] + g.sum(1)[None] + 1)
        return linear_sum_assignment((-2 * prob + 5 * bce + 5 * dice).cpu().numpy())

    def query_loss(out, batch, cfg):
        Lc = len(cfg["LESION_CLASSES"]); dev = out["dense"].device
        H, W = batch["y"].shape[-2:]
        total = 0.0
        n_gt = max(sum(len(l) for l in batch["inst_labels"]), 1)
        for cls_logit, mask_logit in out["aux"]:
            B, K, _ = cls_logit.shape
            tgt = torch.full((B, K), Lc, dtype=torch.long, device=dev)
            lb, ld = 0.0, 0.0
            h2, w2 = mask_logit.shape[-2:]
            for b in range(B):
                gl = batch["inst_labels"][b].to(dev)
                if len(gl) == 0:
                    continue
                gm = batch["inst_masks"][b].to(dev).float()
                gm_lr = F.interpolate(gm[None], size=(h2, w2), mode="area")[0]
                qi, gi = hungarian(cls_logit[b], mask_logit[b], gl, gm_lr)
                qi = torch.as_tensor(qi, device=dev); gi = torch.as_tensor(gi, device=dev)
                tgt[b, qi] = gl[gi]
                pm = F.interpolate(mask_logit[b, qi][None].float(), size=(H, W),
                                   mode="bilinear", align_corners=False)[0]
                gt = gm[gi]
                lb = lb + F.binary_cross_entropy_with_logits(pm, gt, reduction="none").mean((1, 2)).sum()
                s = pm.sigmoid()
                ld = ld + (1 - (2 * (s * gt).sum((1, 2)) + 1) / (s.sum((1, 2)) + gt.sum((1, 2)) + 1)).sum()
            wcls = torch.ones(Lc + 1, device=dev); wcls[-1] = 0.1
            lcls = F.cross_entropy(cls_logit.float().flatten(0, 1), tgt.flatten(), weight=wcls)
            total = total + 2 * lcls + (5 * lb + 5 * ld) / n_gt
        return total / len(out["aux"])

    def tv_loss(alpha):
        if alpha is None:
            return 0.0
        a = alpha.float()
        return (a[..., 1:, :] - a[..., :-1, :]).abs().mean() + (a[..., :, 1:] - a[..., :, :-1]).abs().mean()

    def total_loss(out, batch, cfg):
        l = dense_loss(out["dense"], batch["y"], len(cfg["CLASSES"]), cfg["CE_WEIGHTS"])
        parts = dict(dense=float(l))
        if cfg["QUERY"]:
            lh = focal_hm(out["hm"], batch["hm"]); lq = query_loss(out, batch, cfg)
            l = l + cfg["W_HM"] * lh + cfg["W_QUERY"] * lq
            parts.update(hm=float(lh), query=float(lq))
        if out.get("alpha") is not None:
            l = l + cfg["W_TV"] * tv_loss(out["alpha"])
        return l, parts

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def predict_volume(model, img, cfg, tta=True):
        """Slide a 2.5-D window over every slice of one ROI volume; return class probabilities."""
        dev = next(model.parameters()).device
        model.eval()
        vol = np.asarray(img, np.float32)
        Z, S = vol.shape[0], cfg["N_SLICES"]
        probs = np.zeros((len(cfg["CLASSES"]),) + vol.shape, np.float16)
        bs = 16
        for z0 in range(0, Z, bs):
            zs = range(z0, min(Z, z0 + bs))
            xb = np.stack([vol[np.clip(np.arange(z - S // 2, z + S // 2 + 1), 0, Z - 1)] for z in zs])
            xt = torch.from_numpy(xb).to(dev)
            with torch.autocast(device_type=dev.type, enabled=cfg.get("AMP", False) and dev.type == "cuda"):
                p = fuse(model(xt), cfg)
                if tta:
                    p = 0.5 * (p + fuse(model(xt.flip(-1)), cfg).flip(-1))
            probs[:, z0:z0 + len(zs)] = p.float().permute(1, 0, 2, 3).cpu().numpy()
        return probs

    def organ_gate(lab, cfg, min_ml=2.0, vox=1.0, dilate=3):
        """Kidney neighbourhood implied by the PREDICTION (kidney U predicted lesions), dilated.

        FIX (action-plan 4.3): the legacy gate was `lab == KIDNEY_CLASS`, which excludes voxels
        already predicted as lesion and can therefore delete legitimate components. Including the
        predicted lesions keeps the gate a pure function of the prediction while never starving it.
        """
        organ = np.isin(lab, [cfg["KIDNEY_CLASS"], *cfg["LESION_CLASSES"]])
        cc, n = ndi.label(organ)
        if n:
            sizes = ndi.sum(organ, cc, range(1, n + 1)) * vox / 1000.0
            keep = 1 + np.where(sizes >= min_ml)[0]
            if len(keep):
                organ = np.isin(cc, keep)
        return ndi.binary_dilation(organ, iterations=dilate)

    def postprocess(probs, spacing, cfg):
        """Argmax -> drop small/low-confidence lesions -> keep only lesions near the organ."""
        lab = probs.argmax(0).astype(np.uint8)
        vox = float(np.prod(spacing))
        non_les = [c for c in range(len(cfg["CLASSES"])) if c not in cfg["LESION_CLASSES"]]
        fallback = np.array(non_les)[np.argmax(probs[non_les], 0)]
        gate = organ_gate(lab, cfg, vox=vox) if cfg["KIDNEY_CLASS"] is not None else None
        for c in cfg["LESION_CLASSES"]:
            cc, n = ndi.label(lab == c)
            if n == 0:
                continue
            ids = np.arange(1, n + 1)
            size = ndi.sum(np.ones_like(cc), cc, ids) * vox
            pmax = ndi.maximum(probs[c].astype(np.float32), cc, ids)
            bad = (size < cfg["MIN_LESION_MM3"]) | (pmax < cfg["TAU_HIGH"])
            if gate is not None:
                bad |= ndi.maximum(gate.astype(np.uint8), cc, ids) == 0
            rm = np.isin(cc, ids[bad])
            lab[rm] = fallback[rm]
        return lab

    # ------------------------------------------------------------------ metric primitives
    def dice_mask(a, b):
        sa, sb = int(a.sum()), int(b.sum())
        return 1.0 if sa + sb == 0 else 2.0 * float((a & b).sum()) / (sa + sb)

    def _surface(m):
        st = ndi.generate_binary_structure(m.ndim, 1)
        if m.shape[0] == 1:
            st = st.copy(); st[0] = False; st[2] = False
        return m & ~ndi.binary_erosion(m, structure=st, border_value=1)

    def surf_metrics(pred, gt, spacing, tol=None):
        """HD95 and Normalised Surface Dice in mm. Returns (HD95, NSD, NSD@tol) with tol default 2 mm."""
        tol = CFG["NSD_TOL_MM"] if tol is None else tol
        if pred.sum() == 0 or gt.sum() == 0:
            return np.nan, np.nan
        sp_, sg_ = _surface(pred), _surface(gt)
        if sp_.sum() == 0 or sg_.sum() == 0:
            return np.nan, np.nan
        dg = ndi.distance_transform_edt(~sg_, sampling=spacing)
        dp = ndi.distance_transform_edt(~sp_, sampling=spacing)
        d1, d2 = dg[sp_], dp[sg_]
        hd95 = float(np.percentile(np.concatenate([d1, d2]), 95))
        nsd = float(((d1 <= tol).sum() + (d2 <= tol).sum()) / max(len(d1) + len(d2), 1))
        return hd95, nsd

    def hec_masks(lab, cfg):
        """Official KiTS hierarchical evaluation classes."""
        return {"kidney+masses": lab > 0, "masses": np.isin(lab, [2, 3]),
                "tumor": lab == 2, "cyst": lab == 3}

    def lesion_table(pred, gt, spacing, cfg, key, iou_match=None):
        """Lesion-level TP/FP/FN with deterministic one-to-one matching per class.

        Reviewer 1.7: the legacy metric counted a reference lesion as found when ANY predicted
        lesion voxel overlapped it, class-agnostic. Here we build 3-D components per class, match
        them one-to-one at a declared IoU by optimal assignment, and additionally report the
        class-agnostic "any overlap" variant so the difference is visible in the supplement.
        Diameters are equivalent spherical diameters from the component volume, stated as such.
        """
        iou_match = CFG["LESION_IOU_MATCH"] if iou_match is None else float(iou_match)
        vox = float(np.prod(spacing))
        g_any = np.isin(gt, cfg["LESION_CLASSES"])
        p_any = np.isin(pred, cfg["LESION_CLASSES"])
        rows = []
        for c in cfg["LESION_CLASSES"]:
            gcc, gn = ndi.label(gt == c)
            pcc, pn = ndi.label(pred == c)
            g_ids, p_ids = list(range(1, gn + 1)), list(range(1, pn + 1))
            # O(n_voxels) intersection matrix via a single bincount - avoids materialising masks
            inter = np.zeros((gn, pn), np.float64)
            if gn and pn:
                both = (gcc > 0) & (pcc > 0)
                flat = np.bincount(gcc[both].astype(np.int64) * (pn + 1) + pcc[both].astype(np.int64),
                                   minlength=(gn + 1) * (pn + 1))
                inter = flat.reshape(gn + 1, pn + 1)[1:, 1:]
            gs = np.bincount(gcc.ravel(), minlength=gn + 1)[1:].astype(np.float64)
            ps = np.bincount(pcc.ravel(), minlength=pn + 1)[1:].astype(np.float64)
            iou = inter / np.clip(gs[:, None] + ps[None, :] - inter, 1, None) if (gn and pn) else np.zeros((gn, pn))
            matches = {}
            if gn and pn:
                ri, ci = linear_sum_assignment(-iou)
                for a, b in zip(ri, ci):
                    if iou[a, b] >= iou_match:
                        matches[g_ids[a]] = p_ids[b]
            matched_p = set(matches.values())
            for gid in g_ids:
                vol_mm3 = float(gs[gid - 1]) * vox
                rows.append(dict(key=key, cls=cfg["CLASSES"][c], gt_id=gid, pred_id=matches.get(gid, 0),
                                 matched=int(gid in matches),
                                 diam_mm=round(float((6 * vol_mm3 / math.pi) ** (1 / 3)), 3),
                                 vol_mm3=round(vol_mm3, 1),
                                 iou=round(float(iou[gid - 1, p_ids.index(matches[gid])]) if gid in matches else 0.0, 4),
                                 hit_any_class=bool((p_any & (gcc == gid)).any())))
            for pid in p_ids:
                rows.append(dict(key=key, cls=cfg["CLASSES"][c], gt_id=0, pred_id=pid,
                                 matched=int(pid in matched_p), diam_mm=0.0,
                                 vol_mm3=round(float(ps[pid - 1]) * vox, 1), iou=0.0,
                                 hit_any_class=bool((g_any & (pcc == pid)).any())))
        return rows

    # ------------------------------------------------------- ROI-space scoring
    def score_roi(pred_lab, gt_lab, spacing, cfg, key):
        """All metrics for one ROI volume. Same code path for dev selection and final test."""
        P, G = hec_masks(pred_lab, cfg), hec_masks(gt_lab, cfg)
        r = dict(key=key, case=key.rsplit("_s", 1)[0])
        for name in G:
            r[f"dice_{name}"] = dice_mask(P[name], G[name])
            hd, nsd = surf_metrics(P[name], G[name], spacing)
            r[f"hd95_{name}"] = hd; r[f"nsd_{name}"] = nsd
            r[f"tp_{name}"] = int((P[name] & G[name]).sum())
            r[f"ps_{name}"] = int(P[name].sum()); r[f"gs_{name}"] = int(G[name].sum())
        lrows = lesion_table(pred_lab, gt_lab, spacing, cfg, key)
        fp = sum(1 for x in lrows if x["gt_id"] == 0 and not x["matched"])
        fn = sum(1 for x in lrows if x["gt_id"] > 0 and not x["matched"])
        tp = sum(1 for x in lrows if x["gt_id"] > 0 and x["matched"])
        r.update(fp_lesions=int(fp), tp_lesions=int(tp), fn_lesions=int(fn))
        return r, lrows

    def summarise(case_rows, les_rows, cfg):
        """Fold-level summary; case-level Dice pools both kidney ROIs before averaging."""
        dfc = pd.DataFrame(case_rows); dfl = pd.DataFrame(les_rows)
        s = {}
        agg = dfc.groupby("case")[[c for c in dfc if c.startswith(("tp_", "ps_", "gs_"))]].sum()
        for g in ["kidney+masses", "masses", "tumor", "cyst"]:
            den = (agg[f"ps_{g}"] + agg[f"gs_{g}"]).replace(0, np.nan)
            s[f"Dice {g}"] = float((2 * agg[f"tp_{g}"] / den).fillna(1.0).mean())
        s["FP/ROI"] = float(dfc["fp_lesions"].mean())
        if len(dfl):
            gt = dfl[dfl["gt_id"] > 0].copy()
            s["Sens all"] = float(gt["matched"].mean()) if len(gt) else np.nan
            s["Sens anyoverlap"] = float(gt["hit_any_class"].mean()) if len(gt) else np.nan
            gt["bin"] = pd.cut(gt["diam_mm"], CFG["SIZE_BINS_MM"], right=False)
            small = gt[gt["bin"].cat.codes == 0]
            s["Sens<10mm"] = float(small["matched"].mean()) if len(small) else np.nan
            pred = dfl[dfl["gt_id"] == 0]
            tp_, fp_ = int(gt["matched"].sum()), int(pred["matched"].sum())
            s["Lesion precision"] = tp_ / max(tp_ + len(pred) - fp_, 1)
            s["Lesion F1"] = 2 * tp_ / max(2 * tp_ + len(pred) - fp_ + int((gt["matched"] == 0).sum()), 1)   # == 0, not ~ : bitwise NOT on an int column gives -1/-2
        return s
