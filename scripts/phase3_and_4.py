#!/usr/bin/env python3
import os
import sys
import json
import pickle

# ─── CONFIG ───────────────────────────────────────────────────────────────────────
BASE_DIR    = "/Users/akshat/Data/UIUC/Spring 2025/Courses/CS 568 User-Centered Machine Learning/Project/WebUI-7k"
TRAIN_DIR   = os.path.join(BASE_DIR, "train_split_web7k")
PKL_PATH    = os.path.join(BASE_DIR, "intermediate", "per_page.pkl")
OUTPUT_DIR  = os.path.join(BASE_DIR, "json_dataset_for_agents")

# ─── SETUP ────────────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── LOAD PHASE 1 METADATA ───────────────────────────────────────────────────────
try:
    with open(PKL_PATH, "rb") as f:
        per_page = pickle.load(f)
except Exception as e:
    print(f"❌ Fatal: could not load phase 1 pickle at {PKL_PATH}: {e}", file=sys.stderr)
    sys.exit(1)

# ─── PHASE 3+4: FILTER & MERGE ─────────────────────────────────────────────────────
for page_id, page_data in per_page.items():
    vps = page_data.get("viewports", [])
    out_viewports = []

    for vp_entry in vps:
        vp = vp_entry.get("viewport")
        # reconstruct the same filename prefix you used in phase 1:
        if vp in ("iPad-Pro", "iPhone-13 Pro"):
            prefix = vp
        else:
            prefix = f"default_{vp}"

        axe_path = os.path.join(TRAIN_DIR, page_id, f"{prefix}-axe.json")
        if not os.path.isfile(axe_path):
            print(f"⚠️  Missing axe file, skipping: {axe_path}", file=sys.stderr)
            continue

        # load the axe results
        try:
            with open(axe_path, "r", encoding="utf-8") as axf:
                axe_data = json.load(axf)
        except Exception as e:
            print(f"⚠️  Could not parse JSON {axe_path}: {e}", file=sys.stderr)
            continue

        # keep only if there are real violations
        violations = axe_data.get("violations")
        if isinstance(violations, list) and len(violations) > 0:
            # copy everything from the original vp_entry and inject full axe results
            new_vp = vp_entry.copy()
            new_vp["axe"] = axe_data
            out_viewports.append(new_vp)

    # if at least one viewport had violations, write out the merged JSON
    if out_viewports:
        out_payload = {
            "page_id": page_id,
            "viewports": out_viewports
        }
        out_file = os.path.join(OUTPUT_DIR, f"{page_id}.json")
        try:
            with open(out_file, "w", encoding="utf-8") as outf:
                json.dump(out_payload, outf, indent=2)
            print(f"✅ Wrote {len(out_viewports)} violations → {out_file}")
        except Exception as e:
            print(f"❌ Failed to write {out_file}: {e}", file=sys.stderr)

print("🏁 Phase 3+4 complete.  Check", OUTPUT_DIR)
