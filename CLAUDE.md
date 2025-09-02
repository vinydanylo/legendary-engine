# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a Python-based hackathon project containing a single script for filtering and identifying C# code from mixed-language datasets.

## Key Script

**filter_csharp_scored.py** - Main filtering utility that:
- Identifies C# code using regex patterns for language-specific constructs
- Filters JSON-formatted code datasets to extract C# samples
- Uses scoring system to distinguish C# from Python, JavaScript, and Java
- Processes training and validation datasets separately

## Running the Script

```bash
python filter_csharp_scored.py \
  --in_train input_train.jsonl \
  --in_valid input_valid.jsonl \
  --out_train output_train.jsonl \
  --out_valid output_valid.jsonl
```

### Optional Parameters:
- `--min_csharp_score` (default: 2) - Minimum C# pattern matches required
- `--min_margin` (default: 1) - Minimum score difference between C# and other languages  
- `--others_cap` (default: 2) - Maximum allowed score for other languages
- `--progress_every` (default: 50,000) - Progress reporting interval
- `--max_keep` (default: 0) - Maximum records to keep (0 = unlimited)

## Code Architecture

- **Pattern Detection**: Uses compiled regex patterns to identify language-specific constructs
- **Scoring System**: Each language gets scored based on matched patterns in first 4000 characters
- **Filtering Logic**: C# code must meet minimum score and margin requirements while other languages stay below cap
- **Data Processing**: Processes JSONL files line by line for memory efficiency

## Dependencies

Standard Python libraries only:
- `re`, `json`, `argparse`, `pathlib`, `collections`, `typing`

No external package installation required.