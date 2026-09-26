"""
Machine Learning Model Training & F_0.5 Threshold Calibration
Based on the AWS Builder Center Amazon ML Challenge 2026 guidelines.
Trains a Histogram-based Gradient Boosted Classifier (HistGradientBoosting)
on pairwise string and address features with hard negative mining.
"""

import os
import sys
import time
import re
import joblib
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional
from sklearn.ensemble import HistGradientBoostingClassifier
from rapidfuzz import fuzz

from src.preprocessing import (
    clean_business_name,
    clean_address,
    extract_address_numbers,
    get_core_name_tokens,
    get_name_blocking_keys,
    get_address_blocking_keys,
)
from src.matching import get_num_stems
from src.metrics import evaluate_predictions


def extract_pair_features(cn1: str, ca1: str, nums1: List[str], core1: List[str],
                          cn2: str, ca2: str, nums2: List[str], core2: List[str]) -> List[float]:
    """Extract a 19-dimensional pairwise feature vector for GBDT model."""
    from src.matching import score_pair
    rule = 1.0 if score_pair(cn1, ca1, cn2, ca2, nums1, nums2, core1, core2) else 0.0
    
    name_sort = fuzz.token_sort_ratio(cn1, cn2)
    name_set = fuzz.token_set_ratio(cn1, cn2)
    name_ratio = fuzz.ratio(cn1, cn2)
    name_partial = fuzz.partial_ratio(cn1, cn2)
    
    first_ratio = fuzz.ratio(core1[0], core2[0]) if (core1 and core2) else 0.0
    has_first_match = 1.0 if (first_ratio >= 65 or (core1 and core2 and len(core1[0]) <= 2 and core1[0] == core2[0])) else 0.0
    
    c1_str = " ".join(core1)
    c2_str = " ".join(core2)
    core_sort = fuzz.token_sort_ratio(c1_str, c2_str)
    
    compact1 = re.sub(r"[^a-z0-9]", "", cn1)
    compact2 = re.sub(r"[^a-z0-9]", "", cn2)
    compact_sub = 1.0 if (len(compact1) >= 4 and len(compact2) >= 4 and 
                         (compact1 in compact2 or compact2 in compact1)) else 0.0
    
    stems1 = get_num_stems(nums1)
    stems2 = get_num_stems(nums2)
    num_match = 1.0 if (stems1 and stems2 and bool(stems1 & stems2)) else 0.0
    num_conflict = 1.0 if (stems1 and stems2 and not bool(stems1 & stems2)) else 0.0
    
    has_addr1 = 1.0 if ca1 else 0.0
    has_addr2 = 1.0 if ca2 else 0.0
    both_addr = 1.0 if (ca1 and ca2) else 0.0
    
    addr_sort = fuzz.token_sort_ratio(ca1, ca2) if (has_addr1 and has_addr2) else 0.0
    addr_set = fuzz.token_set_ratio(ca1, ca2) if (has_addr1 and has_addr2) else 0.0
    addr_partial = fuzz.partial_ratio(ca1, ca2) if (has_addr1 and has_addr2) else 0.0
    
    return [
        rule, float(name_sort), float(name_set), float(name_ratio), float(name_partial),
        float(first_ratio), has_first_match, float(core_sort), compact_sub,
        num_match, num_conflict, has_addr1, has_addr2, both_addr,
        float(addr_sort), float(addr_set), float(addr_partial),
        float(abs(len(cn1) - len(cn2))), float(abs(len(ca1) - len(ca2)))
    ]


def build_training_dataset(
    train_dir: str,
    num_s1_entities: int = 15000,
    negatives_per_entity: int = 6
):
    """Generate pairwise dataset from ground truth positives and blocking hard negatives."""
    print(f"Loading {num_s1_entities:,} Source 1 entities for training...")
    
    s1_records = {}
    with open(os.path.join(train_dir, "train_source1.tsv"), "r", encoding="utf-8") as f:
        next(f)
        for i, line in enumerate(f):
            if i >= num_s1_entities:
                break
            p = line.rstrip("\r\n").split("\t")
            s1_records[p[0]] = (p[1] if len(p) > 1 else "", p[2] if len(p) > 2 else "", p[3] if len(p) > 3 else "")

    s1_ids = set(s1_records.keys())
    gt_map = {}
    needed_s23 = set()
    with open(os.path.join(train_dir, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if p[0] in s1_ids:
                matches = [x.strip() for x in p[1].split(",") if x.strip()] if len(p) > 1 and p[1] else []
                gt_map[p[0]] = set(matches)
                needed_s23.update(matches)
                if len(gt_map) == len(s1_ids):
                    break

    print(f"Loaded ground truth. Found {len(needed_s23):,} true matching S2/S3 IDs.")

    # Load S2 and S3 records (including true targets + distractors)
    s23_records = {}
    distractors = 0
    for fn in ["train_source2.tsv", "train_source3.tsv"]:
        filepath = os.path.join(train_dir, fn)
        with open(filepath, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.rstrip("\r\n").split("\t")
                eid = p[0]
                if eid in needed_s23:
                    s23_records[eid] = (p[1] if len(p) > 1 else "", p[2] if len(p) > 2 else "", p[3] if len(p) > 3 else "")
                elif distractors < 100000:
                    s23_records[eid] = (p[1] if len(p) > 1 else "", p[2] if len(p) > 2 else "", p[3] if len(p) > 3 else "")
                    distractors += 1

    print(f"Loaded S2/S3 pool: {len(s23_records):,} records.")

    # Build blocking index for negative mining
    print("Building blocking index for hard negative generation...")
    index = defaultdict(list)
    s23_precomputed = {}
    for eid, (name, addr, country) in s23_records.items():
        cn = clean_business_name(name)
        ca = clean_address(addr, country)
        nums = extract_address_numbers(addr)
        core = get_core_name_tokens(cn)
        s23_precomputed[eid] = (cn, ca, nums, core)
        for k in get_name_blocking_keys(cn): index[(country, k)].append(eid)
        for k in get_address_blocking_keys(ca, addr, country): index[(country, k)].append(eid)

    # Split S1 into 80% train / 20% validation
    all_s1_list = list(s1_records.keys())
    split_idx = int(0.8 * len(all_s1_list))
    train_ids = all_s1_list[:split_idx]
    val_ids = all_s1_list[split_idx:]

    max_bucket_size = 500
    print(f"Extracting pairwise features: {len(train_ids):,} train entities, {len(val_ids):,} validation entities...")
    X_train, y_train = [], []
    for sid in train_ids:
        n1, a1, c1 = s1_records[sid]
        cn1 = clean_business_name(n1)
        ca1 = clean_address(a1, c1)
        nums1 = extract_address_numbers(a1)
        core1 = get_core_name_tokens(cn1)
        
        cands = set()
        for k in get_name_blocking_keys(cn1):
            bucket = index.get((c1, k), [])
            if bucket and len(bucket) <= max_bucket_size:
                cands.update(bucket)
        for k in get_address_blocking_keys(ca1, a1, c1):
            bucket = index.get((c1, k), [])
            if bucket and len(bucket) <= max_bucket_size:
                cands.update(bucket)
        
        true_set = gt_map.get(sid, set())
        # Positive pairs
        for mid in true_set:
            if mid in s23_precomputed:
                cn2, ca2, nums2, core2 = s23_precomputed[mid]
                X_train.append(extract_pair_features(cn1, ca1, nums1, core1, cn2, ca2, nums2, core2))
                y_train.append(1)
                
        # Hard negative pairs from blocking
        neg_count = 0
        for cid in cands:
            if cid not in true_set and cid in s23_precomputed:
                cn2, ca2, nums2, core2 = s23_precomputed[cid]
                X_train.append(extract_pair_features(cn1, ca1, nums1, core1, cn2, ca2, nums2, core2))
                y_train.append(0)
                neg_count += 1
                if neg_count >= negatives_per_entity:
                    break

    # Build validation evaluation structures
    val_gt = {sid: gt_map.get(sid, set()) for sid in val_ids}
    val_data = []
    for sid in val_ids:
        n1, a1, c1 = s1_records[sid]
        cn1 = clean_business_name(n1)
        ca1 = clean_address(a1, c1)
        nums1 = extract_address_numbers(a1)
        core1 = get_core_name_tokens(cn1)
        
        cands = set()
        for k in get_name_blocking_keys(cn1):
            bucket = index.get((c1, k), [])
            if bucket and len(bucket) <= max_bucket_size:
                cands.update(bucket)
        for k in get_address_blocking_keys(ca1, a1, c1):
            bucket = index.get((c1, k), [])
            if bucket and len(bucket) <= max_bucket_size:
                cands.update(bucket)
        
        cand_list = []
        for cid in cands:
            if cid in s23_precomputed:
                cn2, ca2, nums2, core2 = s23_precomputed[cid]
                feat = extract_pair_features(cn1, ca1, nums1, core1, cn2, ca2, nums2, core2)
                cand_list.append((cid, feat))
        val_data.append((sid, cand_list))

    return np.array(X_train, dtype=np.float32), np.array(y_train, dtype=np.int32), val_data, val_gt


def train_and_calibrate(train_dir: str, model_save_path: str):
    """Train HistGradientBoosting model and calibrate optimal F_0.5 threshold."""
    X_train, y_train, val_data, val_gt = build_training_dataset(train_dir)
    print(f"\nTraining set size: {len(X_train):,} pairs (Positives: {sum(y_train):,}, Negatives: {len(y_train)-sum(y_train):,})")
    
    print("Fitting HistGradientBoostingClassifier (LightGBM equivalent)...")
    clf = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.04,
        max_leaf_nodes=45,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42
    )
    clf.fit(X_train, y_train)
    print("Model fitting complete!")

    # Threshold calibration on validation set
    print("\nCalibrating decision threshold on validation set:")
    all_val_feats = []
    val_mapping = []
    for sid, cand_list in val_data:
        for cid, feat in cand_list:
            all_val_feats.append(feat)
            val_mapping.append((sid, cid))

    prob_map = defaultdict(dict)
    if all_val_feats:
        probs = clf.predict_proba(np.array(all_val_feats, dtype=np.float32))[:, 1]
        for (sid, cid), p in zip(val_mapping, probs):
            prob_map[sid][cid] = p

    # Baseline comparison: heuristic rule
    h_preds = {}
    for sid, cand_list in val_data:
        h_preds[sid] = {cid for cid, feat in cand_list if feat[0] > 0.5}
    h_res = evaluate_predictions(val_gt, h_preds)
    print(f"  [Baseline Heuristic Rule] -> Macro F0.5: {h_res['macro_f05']:.4f} | Precision: {h_res['mean_precision']:.3f} | Recall: {h_res['mean_recall']:.3f} | SingAcc: {h_res['singleton_accuracy']:.3f}")

    best_thresh = 0.80
    best_f05 = 0.0
    for thresh in np.arange(0.50, 0.99, 0.02):
        preds = {}
        for sid, _ in val_data:
            matched = {cid for cid, p in prob_map[sid].items() if p >= thresh}
            preds[sid] = matched
        res = evaluate_predictions(val_gt, preds)
        print(f"  Threshold {thresh:.2f} -> Macro F0.5: {res['macro_f05']:.4f} | Precision: {res['mean_precision']:.3f} | Recall: {res['mean_recall']:.3f} | SingAcc: {res['singleton_accuracy']:.3f}")
        if res['macro_f05'] > best_f05:
            best_f05 = res['macro_f05']
            best_thresh = float(thresh)

    print(f"\n=======================================================")
    print(f" OPTIMAL THRESHOLD: {best_thresh:.2f} (Validation Macro F_0.5: {best_f05:.4f})")
    print(f"=======================================================")

    # Save model artifact and optimal threshold
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump({"model": clf, "threshold": best_thresh, "val_f05": best_f05}, model_save_path)
    print(f"Saved trained model and calibrated threshold to {model_save_path}")


if __name__ == "__main__":
    train_and_calibrate(
        train_dir="student_resource/dataset/train",
        model_save_path="code/business_entity_resolution/models/model.joblib"
    )
