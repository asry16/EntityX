"""
Pairwise feature scoring and precision-calibrated entity matching.
Optimized for the F_0.5 evaluation metric (precision-weighted, singleton penalty).
"""

import re
from typing import Set, Tuple, List, Dict
from rapidfuzz import fuzz
from src.preprocessing import (
    clean_business_name,
    clean_address,
    extract_address_numbers,
    get_core_name_tokens,
)


def get_num_stems(nums: List[str]) -> Set[str]:
    """Extract base numeric digit stems to align numbers with unit/block suffixes (e.g. 31 from 31D)."""
    stems = set()
    for n in nums:
        stems.add(n)
        d = re.sub(r"[^0-9]", "", n)
        if d:
            stems.add(d)
    return stems


def score_pair(
    cn1: str,
    ca1: str,
    cn2: str,
    ca2: str,
    nums1: List[str],
    nums2: List[str],
    core1: List[str],
    core2: List[str],
) -> bool:
    """Evaluate whether (S1, S_cand) is a genuine entity match.
    
    Optimized for macro F_0.5:
    - High-order token sorting captures transpositions (e.g., PLLC Novent Owl vs. Novent Owl PLLC).
    - Compact alphanumeric substring matching aligns handles/domain names (e.g., tmrindia vs tmr (india)).
    - Numeric stem intersection verifies physical street identity while supporting unit letters (e.g., 31 vs 31D).
    - Prevents false merges on shared commercial addresses by requiring core name compatibility.
    """
    name_sort = fuzz.token_sort_ratio(cn1, cn2)
    name_set = fuzz.token_set_ratio(cn1, cn2)
    partial_name = fuzz.partial_ratio(cn1, cn2)
    
    first_ratio = fuzz.ratio(core1[0], core2[0]) if (core1 and core2) else 0
    has_first_match = (
        first_ratio >= 65 or 
        (core1 and core2 and len(core1[0]) <= 2 and core1[0] == core2[0])
    )
    
    has_addr1 = bool(ca1)
    has_addr2 = bool(ca2)
    
    stems1 = get_num_stems(nums1)
    stems2 = get_num_stems(nums2)
    num_match = bool(stems1 & stems2)
    
    # Compact alphanumeric match (e.g. tmrindia in tmr (india))
    compact1 = re.sub(r"[^a-z0-9]", "", cn1)
    compact2 = re.sub(r"[^a-z0-9]", "", cn2)
    compact_sub = (
        len(compact1) >= 4 and len(compact2) >= 4 and 
        (compact1 in compact2 or compact2 in compact1)
    )
    
    addr_sort = fuzz.token_sort_ratio(ca1, ca2) if (has_addr1 and has_addr2) else 0
    addr_set = fuzz.token_set_ratio(ca1, ca2) if (has_addr1 and has_addr2) else 0

    # Rule 1: Transposed or near-identical name (>= 88) across full tokens
    if name_sort >= 88:
        if has_addr1 and has_addr2:
            if num_match or addr_set >= 60 or addr_sort >= 50:
                return True
        else:
            return True
            
    # Rule 2: Strong name similarity (>= 80) + first token / compact anchor
    if name_sort >= 80 and (has_first_match or compact_sub):
        if has_addr1 and has_addr2:
            if num_match or addr_set >= 60 or addr_sort >= 50:
                return True
        else:
            return True
            
    # Rule 3: Exact physical address match + compatible name / DBA / domain handle
    if has_addr1 and has_addr2:
        if (num_match and addr_sort >= 72) or addr_sort >= 85:
            if partial_name >= 60 or name_set >= 45 or has_first_match or compact_sub:
                return True
                
    return False


def match_entity(
    s1_id: str,
    s1_name: str,
    s1_addr: str,
    country: str,
    candidates: List[str],
    s23_data: Dict[str, Tuple[str, str, List[str], List[str]]],
) -> Tuple[List[str], List[str]]:
    """Match candidates for a single Source 1 entity."""
    if not candidates:
        return [], []
    
    cn1 = clean_business_name(s1_name)
    ca1 = clean_address(s1_addr, country)
    nums1 = extract_address_numbers(s1_addr)
    core1 = get_core_name_tokens(cn1)
    
    matched_ids = []
    for cid in candidates:
        if cid not in s23_data:
            continue
        cn2, ca2, nums2, core2 = s23_data[cid]
        if score_pair(cn1, ca1, cn2, ca2, nums1, nums2, core1, core2):
            matched_ids.append(cid)
            
    return matched_ids, candidates
