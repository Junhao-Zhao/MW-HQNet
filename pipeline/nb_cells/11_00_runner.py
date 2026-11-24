# ============================================================================
# CELL 11 - Experiment runner: what to train, in what order, and why
#
# Under an 8 GB GPU the ordering matters more than the ambition. This cell implements the
# stop-rule ladder from the action plan: everything that can break the paper's claims is
# tested on ONE locked fold first, and only then is the five-fold sweep launched.
#
#   tier 1  pilot    B0 + B1 on one fold           ~2 h    do the numbers behave?
#   tier 2  decisive A3 vs A4 on the same fold     ~2 h    does the graph beat attention?
#   tier 3  main     B0 + B1 on all five folds     ~10 h   the paper's Table II
#   tier 4  extra    A1, A2, A6, A7, B0_wide       ~10 h   Table IV / Table V
# ============================================================================
if ENABLED["run_all"] and MODEL_NEEDED:
    PILOT_FOLD = CFG["FOLDS"][0]

    RUN_PLAN = [
        # (tier, experiment, folds)
        ("tier1", "B0_baseline", [PILOT_FOLD]),
        ("tier1", "B1_full", [PILOT_FOLD]),
        ("tier2", "A3_no_graph", [PILOT_FOLD]),
        ("tier2", "A4_selfattn", [PILOT_FOLD]),
        ("tier3", "B0_baseline", list(CFG["FOLDS"])),
        ("tier3", "B1_full", list(CFG["FOLDS"])),
        ("tier4", "A1_fixed_multi", [PILOT_FOLD]),
        ("tier4", "A2_wso", [PILOT_FOLD]),
        ("tier4", "A6_no_fusion", [PILOT_FOLD]),
        ("tier4", "A7_no_aifi", [PILOT_FOLD]),
        ("tier4", "B0_wide", [PILOT_FOLD]),
    ]

    def run_plan(plan=None, max_runs=None, dry=False):
        """Train + evaluate everything in the plan; idempotent, so re-running resumes."""
        plan = plan or RUN_PLAN
        done = set()
        for p in DIRS["results"].glob("*/summary.json"):
            d = json.loads(p.read_text())
            done.add((d["experiment"], d["fold"]))
        todo = [(t, e, f) for (t, e, fs) in plan for f in fs if (e, f) not in done]
        est = 55 * len(todo) / 60.0
        print(f"{len(todo)} (experiment, fold) pairs outstanding; "
              f"one pair is ~55 min on this GPU, so roughly {est:.1f} h of GPU time")
        for t, e, f in todo:
            print(f"    [{t}] {e:18s} fold {f}")
        if dry:
            return todo

        t_all = time.time()
        for i, (tier, exp_id, fold) in enumerate(todo):
            if max_runs and i >= max_runs:
                print(f"stopping after {max_runs} runs (max_runs)")
                break
            print(f"\n--- [{i+1}/{len(todo)}] {tier} {exp_id} fold {fold} "
                  f"({(time.time() - t_all) / 60:.0f} min elapsed) ---")
            try:
                train_run(exp_id, fold)
                eval_run(exp_id, fold)
            except Exception as e:
                import traceback
                print(f"  FAILED {exp_id} fold {fold}: {type(e).__name__}: {e}")
                traceback.print_exc(limit=2)
            if DEV == "cuda":
                torch.cuda.empty_cache()
        return todo

    # ---------------------------------------------------------------- pilot first
    print("Recommended sequence")
    print("  1. run_plan(plan=[('tier1', 'B0_baseline', [PILOT_FOLD]), ('tier1', 'B1_full', [PILOT_FOLD])])")
    print("  2. inspect Cell 13's paired numbers and Cell 11's stop-rule checks")
    print("  3. run_plan(plan=[('tier2', 'A3_no_graph', [PILOT_FOLD]), ('tier2', 'A4_selfattn', [PILOT_FOLD])])")
    print("  4. only then run_plan() for the full ladder")
    print("\nSet ENABLED['run_all']=True and call run_plan(dry=True) to see the cost before committing.")

# ------------------------------------------------------------------ auto-launch
if PROFILE == "pilot" and ENABLED["run_all"]:
    _fold = CFG["FOLDS"][0]
    print("=" * 88)
    print(f"PROFILE = 'pilot'  ->  4 runs on fold {_fold}, roughly 4.5 h on this GPU")
    print("  B0 baseline -> B1 full -> A3 no-graph -> A4 self-attention")
    print("  Leave the browser tab open. Each run writes its own folder under reviewer_work/runs.")
    print("=" * 88, flush=True)
    run_plan(plan=[("tier1", "B0_baseline", [_fold]),
                   ("tier1", "B1_full", [_fold]),
                   ("tier2", "A3_no_graph", [_fold]),
                   ("tier2", "A4_selfattn", [_fold])])
    print("\nPilot finished. Now run the statistics and tables cells further down.")

elif PROFILE == "full5" and ENABLED["run_all"]:
    print("=" * 88)
    print("PROFILE = 'full5'  ->  B0 and B1 on all five folds, roughly 9 h on this GPU")
    print("=" * 88, flush=True)
    run_plan(plan=[("tier3", "B0_baseline", list(CFG["FOLDS"])),
                   ("tier3", "B1_full", list(CFG["FOLDS"]))])
    print("\nFive-fold runs finished. Now run the statistics and tables cells further down.")
