"""
Main CLI entrypoint for Amazon ML Challenge 2026: Business Entity Resolution.
Usage:
    python3 src/main.py --data-dir ../../student_resource/dataset/test --output-dir ../../output
"""

import argparse
import sys
import os

# Add package root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import run_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Amazon ML Challenge 2026: Business Entity Resolution Pipeline"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="student_resource/dataset/test",
        help="Path to directory containing source TSV files (default: student_resource/dataset/test)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save matching_results.tsv and candidate_pairs.tsv (default: output)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="test",
        choices=["test", "train"],
        help="File prefix: 'test' or 'train' (default: test)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional entity limit per country for quick testing/benchmarking",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print(" Amazon ML Challenge 2026: Business Entity Resolution Pipeline")
    print(f" Data Directory:   {args.data_dir}")
    print(f" Output Directory: {args.output_dir}")
    print(f" Mode/Prefix:      {args.prefix}")
    if args.limit:
        print(f" Entity Limit:     {args.limit} per country")
    print("=" * 70)

    run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        prefix=args.prefix,
        limit_per_country=args.limit,
    )


if __name__ == "__main__":
    main()
