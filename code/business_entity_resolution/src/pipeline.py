"""
End-to-End Pipeline for Amazon ML Challenge 2026: Business Entity Resolution.
Processes source data with country partitioning, multi-key inverted indexing,
and precision-calibrated scoring. Outputs matching_results.tsv and candidate_pairs.tsv.
"""

import os
import sys
import gc
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional

from src.preprocessing import (
    clean_business_name,
    clean_address,
    extract_address_numbers,
    get_core_name_tokens,
    get_name_blocking_keys,
    get_address_blocking_keys,
)
from src.matching import score_pair


def stream_source1(s1_path: str, target_country: Optional[str] = None):
    """Yield (s1_id, name, addr, country) from Source 1 TSV."""
    with open(s1_path, "r", encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t")
            eid = parts[0].strip()
            name = parts[1].strip() if len(parts) > 1 else ""
            addr = parts[2].strip() if len(parts) > 2 else ""
            country = parts[3].strip() if len(parts) > 3 else ""
            if target_country and country != target_country:
                continue
            yield eid, name, addr, country


def load_s23_country_index(
    data_dir: str,
    prefix: str,
    country: str,
    max_bucket_size: int = 500
) -> Tuple[Dict[str, list], Dict[str, Tuple[str, str, List[str], List[str]]]]:
    """Build multi-key inverted index and precomputed feature store for a single country.
    
    Reads both prefix_source2.tsv and prefix_source3.tsv for the specified country.
    Returns:
        index: Dict[blocking_key -> List[entity_id]]
        s23_store: Dict[entity_id -> (clean_name, clean_addr, numbers, core_tokens)]
    """
    print(f"[{country}] Building candidate index from Source 2 and Source 3...")
    start_time = time.time()
    
    index = defaultdict(list)
    s23_store = {}
    
    for src_num in ("source2", "source3"):
        filename = f"{prefix}_{src_num}.tsv"
        filepath = os.path.join(data_dir, filename)
        if not os.path.isfile(filepath):
            print(f"Warning: {filepath} not found, skipping.")
            continue
            
        count = 0
        with open(filepath, "r", encoding="utf-8") as f:
            next(f)  # skip header
            for line in f:
                line = line.rstrip("\r\n")
                if not line:
                    continue
                parts = line.split("\t")
                row_country = parts[3].strip() if len(parts) > 3 else ""
                if row_country != country:
                    continue
                    
                eid = parts[0].strip()
                name = parts[1].strip() if len(parts) > 1 else ""
                addr = parts[2].strip() if len(parts) > 2 else ""
                
                cn = clean_business_name(name)
                ca = clean_address(addr, country)
                nums = extract_address_numbers(addr)
                core = get_core_name_tokens(cn)
                
                s23_store[eid] = (cn, ca, nums, core)
                
                # Register blocking keys
                for k in get_name_blocking_keys(cn):
                    index[k].append(eid)
                for k in get_address_blocking_keys(ca, addr, country):
                    index[k].append(eid)
                    
                count += 1
                if count % 1000000 == 0:
                    print(f"  [{country}] Loaded {count:,} records from {filename}...")
                    
        print(f"  [{country}] Finished {filename}: {count:,} records loaded.")
        
    print(f"[{country}] Index built: {len(s23_store):,} entities, {len(index):,} unique keys in {time.time() - start_time:.1f}s.")
    return index, s23_store


def process_country(
    country: str,
    data_dir: str,
    prefix: str,
    matching_f,
    candidate_f,
    max_candidates_per_entity: int = 100,
    max_bucket_size: int = 500,
    limit: Optional[int] = None
):
    """Process all Source 1 records for a given country and stream predictions to file."""
    print(f"\n{'='*70}\nProcessing country: {country}\n{'='*70}")
    
    # 1. Load S2 and S3 index for this country
    index, s23_store = load_s23_country_index(data_dir, prefix, country, max_bucket_size)
    
    # 2. Stream S1 entities and perform blocking + matching
    s1_path = os.path.join(data_dir, f"{prefix}_source1.tsv")
    start_time = time.time()
    processed_count = 0
    total_candidates = 0
    total_matches = 0
    singletons = 0
    
    for s1_id, name, addr, c in stream_source1(s1_path, target_country=country):
        cn1 = clean_business_name(name)
        ca1 = clean_address(addr, country)
        nums1 = extract_address_numbers(addr)
        core1 = get_core_name_tokens(cn1)
        
        # Candidate generation (blocking)
        cands_set = set()
        for k in get_name_blocking_keys(cn1):
            bucket = index.get(k)
            if bucket and len(bucket) <= max_bucket_size:
                cands_set.update(bucket)
        for k in get_address_blocking_keys(ca1, addr, country):
            bucket = index.get(k)
            if bucket and len(bucket) <= max_bucket_size:
                cands_set.update(bucket)
                
        # Limit candidate count per entity to prevent pathological explosion
        cands_list = list(cands_set)
        if len(cands_list) > max_candidates_per_entity:
            cands_list = cands_list[:max_candidates_per_entity]
            
        total_candidates += len(cands_list)
        
        # Matching & scoring
        matched_ids = []
        for cid in cands_list:
            cn2, ca2, nums2, core2 = s23_store[cid]
            if score_pair(cn1, ca1, cn2, ca2, nums1, nums2, core1, core2):
                matched_ids.append(cid)
                
        if matched_ids:
            total_matches += len(matched_ids)
        else:
            singletons += 1
            
        # Write output lines
        matching_f.write(f"{s1_id}\t{','.join(matched_ids)}\n")
        candidate_f.write(f"{s1_id}\t{','.join(cands_list)}\n")
        
        processed_count += 1
        if processed_count % 50000 == 0:
            matching_f.flush()
            candidate_f.flush()
            elapsed = time.time() - start_time
            rate = processed_count / elapsed
            print(
                f"  [{country}] Processed {processed_count:,} S1 entities "
                f"({rate:.0f} ent/s) | Matches: {total_matches:,} | Singletons: {singletons:,} "
                f"| Avg Cands: {total_candidates / processed_count:.1f}"
            )
            
        if limit and processed_count >= limit:
            print(f"Reached limit of {limit} entities for {country}.")
            break
            
    elapsed = time.time() - start_time
    print(
        f"[{country}] Complete: {processed_count:,} entities in {elapsed:.1f}s "
        f"({processed_count / max(1, elapsed):.0f} ent/s)."
    )
    print(
        f"  Total Matches: {total_matches:,} | Singletons: {singletons:,} ({singletons / max(1, processed_count):.1%}) "
        f"| Avg Candidates: {total_candidates / max(1, processed_count):.1f}"
    )
    
    # Cleanup memory
    del index
    del s23_store
    gc.collect()


def run_pipeline(
    data_dir: str,
    output_dir: str,
    prefix: str = "test",
    limit_per_country: Optional[int] = None
):
    """Run the complete Business Entity Resolution pipeline.
    
    Generates matching_results.tsv and candidate_pairs.tsv in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    matching_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")
    
    # Initialize output files with headers
    with open(matching_path, "w", encoding="utf-8") as f_match, \
         open(candidate_path, "w", encoding="utf-8") as f_cand:
         
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        
        # Order of countries to process: France (smallest), US, India
        countries = ["France", "US", "India"] if prefix == "test" else ["US", "India"]
        
        total_start = time.time()
        for country in countries:
            process_country(
                country=country,
                data_dir=data_dir,
                prefix=prefix,
                matching_f=f_match,
                candidate_f=f_cand,
                limit=limit_per_country
            )
            
    print(f"\n{'='*70}")
    print(f"Pipeline finished! Generated outputs in {output_dir}")
    print(f"  matching_results: {matching_path} ({os.path.getsize(matching_path) / 1024 / 1024:.2f} MB)")
    print(f"  candidate_pairs:  {candidate_path} ({os.path.getsize(candidate_path) / 1024 / 1024:.2f} MB)")
    print(f"Total time elapsed: {time.time() - total_start:.1f}s")
    print(f"{'='*70}\n")
