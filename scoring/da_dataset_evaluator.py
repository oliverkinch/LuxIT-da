#!/usr/bin/env python3
"""
Danish Dataset Evaluator with Parallel Processing Support

Evaluates Danish instruction-response pairs using OpenAI's API as judge.
Supports parallel execution via dataset splitting and worker IDs.

Usage:
    # Single worker (all data):
    python da_dataset_evaluator.py --dataset data.jsonl --output results.jsonl

    # Parallel execution (run each in a separate tmux window):
    python da_dataset_evaluator.py --dataset data.jsonl --output results --num-workers 4 --worker-id 0
    python da_dataset_evaluator.py --dataset data.jsonl --output results --num-workers 4 --worker-id 1
    python da_dataset_evaluator.py --dataset data.jsonl --output results --num-workers 4 --worker-id 2
    python da_dataset_evaluator.py --dataset data.jsonl --output results --num-workers 4 --worker-id 3

    # Merge results after all workers complete:
    python da_dataset_evaluator.py --merge --output results --num-workers 4
"""

import json
import time
import os
import random
import argparse
import hashlib
from typing import Dict, Optional
from pathlib import Path
import logging
from datetime import datetime
from dotenv import load_dotenv

import pandas as pd
from openai import OpenAI

load_dotenv()


def setup_logging(worker_id: Optional[int] = None, log_dir: str = "logs") -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"evaluator_worker_{worker_id}" if worker_id is not None else "evaluator")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []

    worker_suffix = f"_worker{worker_id}" if worker_id is not None else ""
    log_file = Path(log_dir) / f"evaluation{worker_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s - Worker{w} - %(levelname)s - %(message)s".format(w=worker_id if worker_id is not None else "Main"),
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(ch)

    logger.info(f"Logging initialised. Log file: {log_file}")
    return logger


def load_dataset(dataset_path: str, logger: logging.Logger) -> pd.DataFrame:
    logger.info(f"Loading dataset from: {dataset_path}")

    if dataset_path.endswith(".csv"):
        df = pd.read_csv(dataset_path)
    elif dataset_path.endswith(".json"):
        df = pd.read_json(dataset_path)
    elif dataset_path.endswith(".jsonl"):
        records = []
        with open(dataset_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping malformed JSON at line {line_num}: {e}")
        df = pd.DataFrame(records)
        logger.info(f"Loaded {len(df)} records from JSONL")
    else:
        raise ValueError(f"Unsupported format: {dataset_path}. Use .csv, .json, or .jsonl")

    required = ["instruction", "response"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    logger.info(f"Dataset loaded: {len(df)} entries, columns: {list(df.columns)}")
    return df


def save_dataset(df: pd.DataFrame, output_path: str, logger: logging.Logger):
    logger.info(f"Saving dataset to: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    path_lower = output_path.lower()
    if path_lower.endswith(".csv"):
        df.to_csv(output_path, index=False, encoding="utf-8")
    elif path_lower.endswith(".json"):
        df.to_json(output_path, orient="records", force_ascii=False, indent=2)
    elif path_lower.endswith(".jsonl"):
        with open(output_path, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                row_dict = {}
                for key, value in row.to_dict().items():
                    if isinstance(value, (list, dict)):
                        row_dict[key] = value
                    elif hasattr(value, "tolist"):
                        row_dict[key] = value.tolist()
                    elif value is None:
                        row_dict[key] = None
                    elif isinstance(value, float) and pd.isna(value):
                        row_dict[key] = None
                    elif isinstance(value, (int, float, str, bool)):
                        row_dict[key] = value
                    else:
                        try:
                            row_dict[key] = None if pd.isna(value) else str(value)
                        except (ValueError, TypeError):
                            row_dict[key] = value
                f.write(json.dumps(row_dict, ensure_ascii=False) + "\n")
    else:
        raise ValueError(f"Unsupported output format: {output_path}")

    logger.info(f"Saved {len(df)} entries to {output_path}")


class DanishDatasetEvaluator:
    """
    Evaluator for Danish instruction-response pairs using OpenAI's API as judge.

    Key features:
    - Checkpoint-based recovery for interrupted processing
    - Parallel processing support with worker IDs
    - Incremental evaluation support
    - Robust retry logic with exponential backoff
    """

    SCORE_COLUMNS = ["linguistic_quality", "factual_accuracy",
                     "instruction_adherence", "helpfulness_relevance"]

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        checkpoint_dir: str = "checkpoints",
        worker_id: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.client    = OpenAI(api_key=api_key)
        self.model     = model
        self.worker_id = worker_id
        self.logger    = logger or logging.getLogger(__name__)

        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        worker_suffix       = f"_worker{worker_id}" if worker_id is not None else ""
        self.checkpoint_file = Path(checkpoint_dir) / f"checkpoint{worker_suffix}.json"

        self.processed_ids, self.checkpoint_scores = self._load_checkpoint()
        self.logger.info(f"Initialised evaluator (model: {model}, worker: {worker_id})")
        self.logger.info(f"Checkpoint file: {self.checkpoint_file}")
        self.logger.info(f"Previously processed entries: {len(self.processed_ids)}")

    # ── Checkpoint management ─────────────────────────────────────────────────

    def _load_checkpoint(self):
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file) as f:
                    data = json.load(f)
                    processed = set(data.get("processed_ids", []))
                    scores    = data.get("scores", {})
                    self.logger.info(f"Loaded checkpoint: {len(processed)} IDs, {len(scores)} scores")
                    return processed, scores
            except Exception as e:
                self.logger.warning(f"Could not load checkpoint: {e}. Starting fresh.")
        return set(), {}

    def _save_checkpoint(self):
        data = {
            "processed_ids": sorted(list(self.processed_ids)),
            "scores":        self.checkpoint_scores,
            "worker_id":     self.worker_id,
            "count":         len(self.processed_ids),
            "last_updated":  datetime.now().isoformat(),
        }
        temp = self.checkpoint_file.with_suffix(".tmp")
        try:
            with open(temp, "w") as f:
                json.dump(data, f)
            temp.rename(self.checkpoint_file)
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            if temp.exists():
                try:
                    temp.unlink()
                except Exception:
                    pass
            raise

    # ── Entry ID ──────────────────────────────────────────────────────────────

    def _generate_entry_id(self, instruction: str, response: str) -> str:
        return "EVAL_" + hashlib.md5(f"{instruction}||{response}".encode("utf-8")).hexdigest()[:16]

    # ── Evaluation prompt ─────────────────────────────────────────────────────

    def _create_evaluation_prompt(self, instruction: str, response: str) -> str:
        return f"""You are an expert evaluator of Danish text quality. Your task is to evaluate the following instruction-response pair written in Danish based on four specific criteria.

IMPORTANT: The texts below are in Danish. You must evaluate them as Danish texts, NOT as Swedish, Norwegian, or any other language. Danish has its own distinct grammar, vocabulary, and spelling conventions.

INSTRUCTION (in Danish):
{instruction}

RESPONSE (in Danish):
{response}

EVALUATION CRITERIA:

1. linguistic_quality (Linguistic Quality):
- Score 1 (Poor): Contains significant grammatical errors, spelling mistakes, or unnatural phrasing in Danish. Text that is actually Swedish or Norwegian instead of proper Danish should receive this score.
- Score 2 (Acceptable): Mostly correct Danish, but has minor errors or sounds slightly robotic/unnatural.
- Score 3 (Excellent): Fluent, idiomatic, and grammatically correct Danish. Natural-sounding text that a native speaker would produce.

2. factual_accuracy (Factual Accuracy):
- Score 1 (Incorrect): Contains factual errors that contradict the source text or general knowledge.
- Score 2 (Mostly Correct): Mostly accurate but might have minor inaccuracies or omissions.
- Score 3 (Perfect): Completely accurate according to the source text and factual knowledge.

3. instruction_adherence (Instruction Following):
- Score 1 (Not Followed): Fails to follow the core instruction (e.g., provides a summary when asked for a list).
- Score 2 (Partially Followed): Follows the main instruction but misses a constraint (e.g., wrong length, format, or tone).
- Score 3 (Fully Followed): Perfectly follows all parts of the instruction, including constraints like length, format, and tone.

4. helpfulness_relevance (Helpfulness and Relevance):
- Score 1 (Not Helpful): The instruction is nonsensical, irrelevant, or the response is unhelpful/off-topic.
- Score 2 (Somewhat Helpful): The instruction is plausible but not very insightful. The response addresses it in a basic way.
- Score 3 (Very Helpful): A genuinely useful, interesting, or creative instruction that elicits a helpful, comprehensive response.

CRITICAL INSTRUCTIONS:
- Respond ONLY with a JSON object containing the four scores.
- Each score must be an integer: 1, 2, or 3.
- Do NOT include any explanations, comments, or additional text outside the JSON.
- Evaluate the text AS DANISH, not as any other language.

JSON FORMAT:
{{
    "linguistic_quality": <score 1-3>,
    "factual_accuracy": <score 1-3>,
    "instruction_adherence": <score 1-3>,
    "helpfulness_relevance": <score 1-3>
}}"""

    # ── API call ──────────────────────────────────────────────────────────────

    def _call_api(
        self,
        instruction: str,
        response: str,
        max_retries: int = 10,
        base_delay: float = 2.0,
        max_delay: float = 120.0,
    ) -> Optional[Dict[str, int]]:
        prompt = self._create_evaluation_prompt(instruction, response)

        for attempt in range(max_retries):
            try:
                api_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert evaluator of Danish text quality. "
                                "You understand the distinct characteristics of standard Danish (rigsdansk) "
                                "as separate from Swedish and Norwegian. "
                                "Always respond with valid JSON only."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=1,
                    response_format={"type": "json_object"},
                )

                content = api_response.choices[0].message.content or ""
                scores = json.loads(content)

                if all(k in scores for k in self.SCORE_COLUMNS):
                    for k in self.SCORE_COLUMNS:
                        if scores[k] not in [1, 2, 3]:
                            raise ValueError(f"Invalid score for {k}: {scores[k]}")
                    return scores
                else:
                    raise ValueError(f"Missing keys in response: {scores}")

            except Exception as e:
                msg = str(e).lower()
                if "rate limit" in msg or "429" in str(e) or "rate_limit" in msg:
                    delay = min(base_delay * (2 ** attempt), max_delay) * (0.5 + random.random())
                    self.logger.warning(f"Rate limit (attempt {attempt + 1}/{max_retries}). Waiting {delay:.1f}s…")
                    time.sleep(delay)
                elif "timeout" in msg or "connection" in msg:
                    delay = min(base_delay * (1.5 ** attempt), 60)
                    self.logger.warning(f"Network issue (attempt {attempt + 1}/{max_retries}): {e}. Waiting {delay:.1f}s…")
                    time.sleep(delay)
                else:
                    self.logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(base_delay * (2 ** attempt))
                    else:
                        self.logger.error(f"Failed after {max_retries} attempts: {e}")
                        return None
        return None

    # ── Dataset evaluation ────────────────────────────────────────────────────

    def evaluate_dataset(
        self,
        dataset: pd.DataFrame,
        output_path: str,
        checkpoint_interval: int = 10,
        save_interval: int = 100,
        rate_limit_delay: float = 0.5,
        force_reevaluate: bool = False,
    ) -> pd.DataFrame:
        result_df = dataset.copy()

        for col in self.SCORE_COLUMNS:
            if col not in result_df.columns:
                result_df[col] = None

        result_df["__eval_id__"] = result_df.apply(
            lambda row: self._generate_entry_id(row["instruction"], row["response"]), axis=1
        )

        # Deduplicate IDs
        id_counts  = result_df["__eval_id__"].value_counts()
        duplicates = id_counts[id_counts > 1]
        if len(duplicates) > 0:
            self.logger.warning(f"Found {len(duplicates)} duplicate instruction-response pairs")
            seen, new_ids = {}, []
            for eval_id in result_df["__eval_id__"]:
                if eval_id in seen:
                    seen[eval_id] += 1
                    new_ids.append(f"{eval_id}_DUP{seen[eval_id]:03d}")
                else:
                    seen[eval_id] = 0
                    new_ids.append(eval_id)
            result_df["__eval_id__"] = new_ids

        # Merge previous results
        if os.path.exists(output_path) and not force_reevaluate:
            try:
                prev_df = load_dataset(output_path, self.logger)
                self.logger.info(f"Found previous results: {len(prev_df)} entries")
                if "__eval_id__" in prev_df.columns:
                    prev_scores = prev_df.set_index("__eval_id__")[self.SCORE_COLUMNS]
                    for idx, row in result_df.iterrows():
                        eid = row["__eval_id__"]
                        if eid in prev_scores.index:
                            prev_row = prev_scores.loc[eid]
                            if prev_row.notna().all():
                                for col in self.SCORE_COLUMNS:
                                    result_df.at[idx, col] = prev_row[col]
                                self.processed_ids.add(eid)
                    self.logger.info(f"Merged {len(self.processed_ids)} previously evaluated entries")
            except Exception as e:
                self.logger.warning(f"Could not load previous results: {e}")

        # Restore from checkpoint
        if self.checkpoint_scores and not force_reevaluate:
            restored = 0
            for idx, row in result_df.iterrows():
                eid = row["__eval_id__"]
                if eid in self.checkpoint_scores:
                    for col in self.SCORE_COLUMNS:
                        if col in self.checkpoint_scores[eid]:
                            result_df.at[idx, col] = self.checkpoint_scores[eid][col]
                    self.processed_ids.add(eid)
                    restored += 1
            if restored:
                self.logger.info(f"Restored {restored} entries from checkpoint")

        total, processed, skipped, failed = len(result_df), 0, 0, 0
        failed_ids  = []
        start_time  = time.time()

        self.logger.info(f"Starting evaluation of {total} entries… ({len(self.processed_ids)} already done)")

        try:
            for idx, row in result_df.iterrows():
                eid = row["__eval_id__"]

                if eid in self.processed_ids and not force_reevaluate:
                    skipped += 1
                    continue

                if not force_reevaluate:
                    if all(pd.notna(row.get(c)) for c in self.SCORE_COLUMNS):
                        self.processed_ids.add(eid)
                        skipped += 1
                        continue

                scores = self._call_api(row["instruction"], row["response"])

                if scores:
                    for col, val in scores.items():
                        result_df.at[idx, col] = val
                    self.processed_ids.add(eid)
                    self.checkpoint_scores[eid] = scores
                    processed += 1

                    if processed % 10 == 0:
                        elapsed   = time.time() - start_time
                        rate      = processed / elapsed if elapsed > 0 else 0
                        remaining = (total - processed - skipped) / rate if rate > 0 else 0
                        self.logger.info(
                            f"Progress: {processed + skipped}/{total} "
                            f"({(processed + skipped) / total * 100:.1f}%) | "
                            f"Rate: {rate:.2f}/s | ETA: {remaining / 60:.1f}min"
                        )

                    if processed % checkpoint_interval == 0:
                        self._save_checkpoint()
                    if processed % save_interval == 0:
                        self._save_results(result_df, output_path)
                        self.logger.info(f"Results saved ({processed} new entries)")
                else:
                    failed += 1
                    failed_ids.append(eid)
                    self.logger.error(f"Failed to evaluate entry: {eid[:20]}…")

                if rate_limit_delay > 0:
                    time.sleep(rate_limit_delay * (0.8 + 0.4 * random.random()))

        except KeyboardInterrupt:
            self.logger.warning("Interrupt received — saving progress…")
            self._save_checkpoint()
            self._save_results(result_df, output_path)
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            try:
                self._save_checkpoint()
                self._save_results(result_df, output_path)
            except Exception as se:
                self.logger.error(f"Failed to save progress: {se}")
            raise

        self._save_checkpoint()
        self._save_results(result_df, output_path)

        elapsed = time.time() - start_time
        self.logger.info(f"""
========================================
Evaluation Complete (Worker {self.worker_id})
========================================
Total entries:     {total}
Newly processed:   {processed}
Skipped:           {skipped}
Failed:            {failed}
Time elapsed:      {elapsed / 60:.1f} minutes
Average rate:      {processed / elapsed if elapsed > 0 else 0:.2f} entries/sec
Output saved to:   {output_path}
========================================""")

        if failed_ids:
            self.logger.warning(f"Failed IDs: {failed_ids[:10]}{'…' if len(failed_ids) > 10 else ''}")

        return result_df.drop(columns=["__eval_id__"])

    def _save_results(self, df: pd.DataFrame, output_path: str):
        path      = Path(output_path)
        temp_path = str(path.parent / f"{path.stem}.tmp{path.suffix}")
        bak_path  = str(path.parent / f"{path.stem}.bak{path.suffix}")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            save_dataset(df.copy(), temp_path, self.logger)
            if os.path.exists(output_path):
                if os.path.exists(bak_path):
                    os.remove(bak_path)
                os.rename(output_path, bak_path)
            os.rename(temp_path, output_path)
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            raise


# ==================== Parallel helpers ====================

def merge_worker_results(
    output_base: str,
    num_workers: int,
    output_format: str = "jsonl",
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    logger = logger or logging.getLogger(__name__)
    logger.info(f"Merging results from {num_workers} workers…")

    all_dfs = []
    for i in range(num_workers):
        for ext in [".jsonl", ".csv", ".json"]:
            worker_path = f"{output_base}_worker{i}{ext}"
            if os.path.exists(worker_path):
                df = load_dataset(worker_path, logger)
                df["__worker_id__"] = i
                all_dfs.append(df)
                logger.info(f"Loaded worker {i}: {len(df)} entries")
                break
        else:
            logger.warning(f"No results found for worker {i}")

    if not all_dfs:
        raise ValueError("No worker results found!")

    merged = pd.concat(all_dfs, ignore_index=True).drop(columns=["__worker_id__"])

    if "__eval_id__" in merged.columns:
        before = len(merged)
        merged = merged.drop_duplicates(subset=["__eval_id__"], keep="first").drop(columns=["__eval_id__"])
        if len(merged) < before:
            logger.info(f"Removed {before - len(merged)} duplicates")

    merged_path = f"{output_base}_merged.{output_format}"
    save_dataset(merged, merged_path, logger)

    evaluated = merged[DanishDatasetEvaluator.SCORE_COLUMNS].notna().all(axis=1).sum()
    logger.info(f"Merge complete — {len(merged)} total entries, {evaluated} fully evaluated → {merged_path}")

    if evaluated > 0:
        for col in DanishDatasetEvaluator.SCORE_COLUMNS:
            dist = merged[col].value_counts().sort_index()
            logger.info(f"  {col}: {dict(dist)}")

    return merged


def generate_tmux_commands(
    dataset_path: str,
    output_base: str,
    num_workers: int,
    model: str = "gpt-4o-mini",
    rate_limit_delay: float = 0.5,
) -> str:
    script = "da_dataset_evaluator.py"
    lines  = [
        f"# Parallel evaluation — {num_workers} workers",
        "tmux new-session -d -s daeval",
    ]
    for i in range(num_workers):
        cmd = (
            f"python {script} "
            f"--dataset {dataset_path} "
            f"--output {output_base}_worker{i}.jsonl "
            f"--num-workers {num_workers} --worker-id {i} "
            f"--model {model} --rate-limit-delay {rate_limit_delay}"
        )
        if i == 0:
            lines.append(f"tmux send-keys -t daeval '{cmd}' Enter")
        else:
            lines.append("tmux new-window -t daeval")
            lines.append(f"tmux send-keys -t daeval '{cmd}' Enter")
    lines.append(f"# After completion: python {script} --merge --output {output_base} --num-workers {num_workers}")
    return "\n".join(lines)


# ==================== CLI ====================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Danish instruction-response pairs using OpenAI API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--merge",         action="store_true", help="Merge results from workers")
    parser.add_argument("--generate-tmux", action="store_true", help="Print tmux launch commands")
    parser.add_argument("--dataset",       type=str,            help="Path to input dataset")
    parser.add_argument("--output",        type=str, required=True, help="Output path / base")
    parser.add_argument("--num-workers",   type=int, default=1)
    parser.add_argument("--worker-id",     type=int, default=None)
    parser.add_argument("--model",         type=str, default="gpt-4o-mini")
    parser.add_argument("--api-key",       type=str, default=None)
    parser.add_argument("--rate-limit-delay",   type=float, default=0.5)
    parser.add_argument("--checkpoint-interval", type=int,   default=10)
    parser.add_argument("--save-interval",       type=int,   default=100)
    parser.add_argument("--force",         action="store_true", help="Re-evaluate all entries")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir",        type=str, default="logs")

    args   = parser.parse_args()
    logger = setup_logging(args.worker_id, args.log_dir)

    if args.generate_tmux:
        if not args.dataset:
            parser.error("--generate-tmux requires --dataset")
        print(generate_tmux_commands(args.dataset, args.output, args.num_workers,
                                     model=args.model, rate_limit_delay=args.rate_limit_delay))
        return

    if args.merge:
        merge_worker_results(args.output, args.num_workers, output_format="jsonl", logger=logger)
        return

    if not args.dataset:
        parser.error("--dataset is required for evaluation")

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        parser.error("OpenAI API key required. Set OPENAI_API_KEY or use --api-key")

    df = load_dataset(args.dataset, logger)

    if args.num_workers > 1:
        if args.worker_id is None:
            parser.error("--worker-id required when --num-workers > 1")
        chunk   = len(df) // args.num_workers
        start   = args.worker_id * chunk
        end     = start + chunk if args.worker_id < args.num_workers - 1 else len(df)
        df      = df.iloc[start:end].copy()
        logger.info(f"Worker {args.worker_id}: entries {start}–{end - 1} ({len(df)} entries)")

    evaluator = DanishDatasetEvaluator(
        api_key=api_key,
        model=args.model,
        checkpoint_dir=args.checkpoint_dir,
        worker_id=args.worker_id,
        logger=logger,
    )

    output_path = args.output
    if not any(output_path.endswith(ext) for ext in [".csv", ".json", ".jsonl"]):
        output_path += ".jsonl"

    evaluator.evaluate_dataset(
        df,
        output_path=output_path,
        checkpoint_interval=args.checkpoint_interval,
        save_interval=args.save_interval,
        rate_limit_delay=args.rate_limit_delay,
        force_reevaluate=args.force,
    )


if __name__ == "__main__":
    main()
