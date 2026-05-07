# -*- coding: utf-8 -*-
"""
Danish Synthetic Data Generator

Generates Danish instruction-response pairs from four seed sources:
  - Danish Wikipedia       (Data/danish_wikipedia.json)
  - Danmarks Statistik     (Data/danmarks_statistik.json)
  - DynaWord               (Data/dynaword.json)
  - Tidsskrift-DK          (Data/tidsskrift.json)

Run fetch_seed_data.py first to download the source data.

Requires INFERENCE_API_KEY in .env (OpenAI-compatible endpoint).
Default model: qwen-235b — override with --model.

Dependencies:
    pip install openai pandas tqdm datasets python-dotenv

Usage:
    python da_synthetic_data_generation.py                        # all sources
    python da_synthetic_data_generation.py --wikipedia-only
    python da_synthetic_data_generation.py --sources wikipedia,dynaword --max-wikipedia 500
    python da_synthetic_data_generation.py --model gemma-3-27b-it
    python da_synthetic_data_generation.py --combine-only
"""

import json
import os
import re
import sys
import time
import logging
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Boilerplate filter (shared with fetch_seed_data.py) ───────────────────────
_BOILERPLATE_SIGNALS = [
    "accept cookies",
    "cookie policy",
    "vi bruger cookies",
    "javascript is required",
    "javascript er påkrævet",
    "enable javascript",
    "you need to enable javascript",
    "dette website anvender cookies",
    "403 forbidden",
    "404 not found",
    "access denied",
]

def _is_boilerplate(text: str) -> bool:
    lower = text.lower()
    return any(signal in lower for signal in _BOILERPLATE_SIGNALS)

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()


# ==================== Data Source Types ====================

class DataSource(Enum):
    WIKIPEDIA          = "wikipedia"
    DANMARKS_STATISTIK = "danmarks_statistik"
    DYNAWORD           = "dynaword"
    TIDSSKRIFT         = "tidsskrift"


# ==================== Configuration ====================

@dataclass
class SourceConfig:
    """Configuration for a single data source."""
    input_file: str
    responses_output_file: str
    final_json_output: str
    final_csv_output: str
    generation_state_file: str
    source_type: DataSource

    def resolve_paths(self, base_path: str):
        self.input_file            = os.path.join(base_path, self.input_file)
        self.responses_output_file = os.path.join(base_path, self.responses_output_file)
        self.final_json_output     = os.path.join(base_path, self.final_json_output)
        self.final_csv_output      = os.path.join(base_path, self.final_csv_output)
        self.generation_state_file = os.path.join(base_path, self.generation_state_file)


@dataclass
class Config:
    """Configuration settings for the Danish data generator."""

    base_path: str = "./"

    # ── Source configs ──────────────────────────────────────────────────────
    wikipedia_config: SourceConfig = field(default_factory=lambda: SourceConfig(
        input_file="Data/danish_wikipedia.json",
        responses_output_file="model_responses_wikipedia.jsonl",
        final_json_output="instruction_answers_wikipedia.json",
        final_csv_output="instruction_answers_wikipedia.csv",
        generation_state_file="generation_state_wikipedia.json",
        source_type=DataSource.WIKIPEDIA,
    ))
    danmarks_statistik_config: SourceConfig = field(default_factory=lambda: SourceConfig(
        input_file="Data/danmarks_statistik.json",
        responses_output_file="model_responses_danmarks_statistik.jsonl",
        final_json_output="instruction_answers_danmarks_statistik.json",
        final_csv_output="instruction_answers_danmarks_statistik.csv",
        generation_state_file="generation_state_danmarks_statistik.json",
        source_type=DataSource.DANMARKS_STATISTIK,
    ))
    dynaword_config: SourceConfig = field(default_factory=lambda: SourceConfig(
        input_file="Data/dynaword.json",
        responses_output_file="model_responses_dynaword.jsonl",
        final_json_output="instruction_answers_dynaword.json",
        final_csv_output="instruction_answers_dynaword.csv",
        generation_state_file="generation_state_dynaword.json",
        source_type=DataSource.DYNAWORD,
    ))
    tidsskrift_config: SourceConfig = field(default_factory=lambda: SourceConfig(
        input_file="Data/tidsskrift.json",
        responses_output_file="model_responses_tidsskrift.jsonl",
        final_json_output="instruction_answers_tidsskrift.json",
        final_csv_output="instruction_answers_tidsskrift.csv",
        generation_state_file="generation_state_tidsskrift.json",
        source_type=DataSource.TIDSSKRIFT,
    ))

    # ── Combined output ─────────────────────────────────────────────────────
    combined_output_json: str = "instruction_tuning_dataset_all_sources.json"
    combined_output_csv:  str = "instruction_tuning_dataset_all_sources.csv"

    # ── Logging ─────────────────────────────────────────────────────────────
    log_file: str = "synthetic_data_generation.log"

    # ── API ─────────────────────────────────────────────────────────────────
    api_key:      str = field(default_factory=lambda: os.environ.get("INFERENCE_API_KEY", ""))
    api_base_url: str = "https://inference.projects.alexandrainst.dk/v1"
    model_name:   str = "qwen-235b"
    max_tokens:   int = 8192

    # ── Generation ──────────────────────────────────────────────────────────
    num_pairs_per_article: int = 3   # maximum; actual count scales with article length
    max_source_chars:      int = 3000  # truncate article text beyond this before sending to LLM
    max_samples_per_source:         Optional[int] = None
    max_wikipedia_samples:          Optional[int] = None
    max_danmarks_statistik_samples: Optional[int] = None
    max_dynaword_samples:           Optional[int] = None
    max_tidsskrift_samples:         Optional[int] = None
    rate_limit_delay: float = 0.5
    max_retries:      int   = 3
    retry_delay:      float = 2.0

    def __post_init__(self):
        self.wikipedia_config.resolve_paths(self.base_path)
        self.danmarks_statistik_config.resolve_paths(self.base_path)
        self.dynaword_config.resolve_paths(self.base_path)
        self.tidsskrift_config.resolve_paths(self.base_path)
        self.combined_output_json = os.path.join(self.base_path, self.combined_output_json)
        self.combined_output_csv  = os.path.join(self.base_path, self.combined_output_csv)
        self.log_file             = os.path.join(self.base_path, self.log_file)
        Path(self.base_path).mkdir(parents=True, exist_ok=True)


# ==================== Logging ====================

def setup_logging(log_file: str) -> logging.Logger:
    logger = logging.getLogger("DaDataGenerator")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ==================== Prompt Template ====================

INSTRUCTION_PROMPT_TEMPLATE = """You are an expert in the Danish language tasked with creating high-quality synthetic training data for language models.

OBJECTIVE:
Generate {num_pairs} instruction-response pair(s) in authentic Danish based on the provided text. These pairs will be used for instruction fine-tuning of language models.

REQUIREMENTS:
1. LANGUAGE: All content MUST be in fluent, natural Danish
   - Use proper Danish grammar, spelling, and idioms
   - Ensure the language sounds natural to native speakers

2. QUALITY STANDARDS:
   - Instructions should be clear, specific, and answerable based on the provided text
   - Responses MUST be substantive: at least 3–4 sentences. Never give a one-sentence response.
   - Responses should be comprehensive, accurate, and well-structured
   - Include ALL necessary context in the instruction for a complete answer
   - If insufficient information exists, indicate that more details are needed

3. SELF-CONTAINED INSTRUCTIONS:
   - Every instruction MUST be fully self-contained. A reader with no other context must be able to answer it.
   - The SOURCE TEXT below is split into two sections: ARTICLE METADATA and ARTICLE TEXT.
   - For text-grounded tasks (summaries, comprehension, "hvad handler teksten om", etc.) paste ONLY
     the content under "ARTICLE TEXT" — never paste the metadata labels. Use natural framing, e.g.:
       "Her er en tekst:\n\n<paste ARTICLE TEXT here>\n\nHvad er den vigtigste pointe?"
       "Kan du opsummere denne tekst kort?\n\n<paste ARTICLE TEXT here>"
       "<paste ARTICLE TEXT here>\n\nHvad handler det her om?"
   - NEVER say "teksten" or "artiklen" without the ARTICLE TEXT being present in the instruction.
   - You MAY use metadata (title, date, journal, authors) to write a standalone question answerable
     from general knowledge, without pasting any text at all.

4. TEMPORAL CONTEXT:
   - When a date is provided, incorporate it appropriately
   - Add temporal context to maintain relevance when applicable

5. DIVERSITY AND NATURAL STYLE:
   - Write instructions exactly as a real Danish user would type them to an AI assistant.
     Use natural, conversational Danish — not formal or academic language.

   - DISTRIBUTION RULE: of the {num_pairs} pairs you generate, AT MOST ONE may paste the article text
     verbatim into the instruction (text-grounded). ALL OTHER pairs must be standalone instructions
     that do NOT include the article text at all, and are answerable from the knowledge contained in
     the article (or from general knowledge). This forces diversity.

   - Standalone instructions MUST be about concepts, definitions, or general-knowledge topics
     that any knowledgeable person could answer without having read the specific article.
     Do NOT write standalone questions about article-specific events, dates, named individuals,
     or unique facts that only appear in this particular article — those require the article text
     to be present to avoid hallucination.
     GOOD standalone: "Hvad er forskellen på moms og lønsumsafgift?"
     BAD standalone:  "Hvornår trådte den ændrede skattepraksis i kraft?" (article-specific fact)

   - Standalone instruction types to use (pick different ones for each pair):
       • Explanation:       "Kan du forklare begrebet X?"  "Hvad betyder X?"
       • Comparison:        "Hvad er forskellen på X og Y?"
       • How-to:            "Hvordan fungerer X?"  "Hvordan beregner man X?"
       • Contextualisation: "Hvad er baggrunden for X?" (general context, not article-specific)
       • Creative reuse:    "Skriv en kort forklaring af X til en 12-årig."
                            "Formuler tre gode spørgsmål om emnet X."

   - Vary opening styles — do NOT start every instruction the same way:
       "Hvad er...?"  /  "Kan du...?"  /  "Jeg vil gerne vide..."  /
       "Forklar mig..."  /  "Hvad handler det om, at...?"  /  "Hvad mener man med...?"

   - Avoid stiff, formal, or test-like language such as "Redegør for", "Analysér", "Beskriv",
     or overly bureaucratic phrasing. Write like a curious person, not a teacher setting an exam.

6. OUTPUT FORMAT:
   Return ONLY a valid JSON array with the following structure:
   [
     {{
       "instruction": "Clear instruction in Danish",
       "response": "Detailed response in Danish"
     }}
   ]

SOURCE TEXT:
{source_context}

Generate {num_pairs} high-quality instruction-response pair(s) based on the above text."""


# ==================== Core Generator ====================

class DanishDataGenerator:
    """Generates synthetic Danish instruction-response pairs from seed articles."""

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.client = self._initialize_client()

    def _initialize_client(self) -> Optional[OpenAI]:
        if not self.config.api_key:
            self.logger.warning("No INFERENCE_API_KEY found in environment.")
            return None
        try:
            client = OpenAI(base_url=self.config.api_base_url, api_key=self.config.api_key)
            self.logger.info("OpenAI-compatible client initialised successfully")
            return client
        except Exception as e:
            self.logger.error(f"Failed to initialise client: {e}")
            return None

    # ── Prompt construction ──────────────────────────────────────────────────

    _DS_ARTIFACT_PATTERNS = [
        # Navigation / UI
        re.compile(r"Vis hele teksten\s*»", re.IGNORECASE),
        re.compile(r"«\s*Minimer teksten", re.IGNORECASE),
        re.compile(r"Del sidens indhold", re.IGNORECASE),
        re.compile(r"Hent som PDF", re.IGNORECASE),
        re.compile(r"Se alle udgivelser", re.IGNORECASE),
        re.compile(r"Næste artikel", re.IGNORECASE),
        re.compile(r"Forrige artikel", re.IGNORECASE),
        re.compile(r"Gå til\b[^\n]*", re.IGNORECASE),
        # Footer metadata block: strip from "Næste udgivelse" or "Alle udgivelser" to end of that line
        re.compile(r"Næste udgivelse:[^\n]*", re.IGNORECASE),
        re.compile(r"Alle udgivelser i serien:[^\n]*", re.IGNORECASE),
        re.compile(r"Statistik\u00addokumentation", re.IGNORECASE),   # soft-hyphen variant
        re.compile(r"Statistikdokumentation", re.IGNORECASE),
        re.compile(r"Kilder og metode", re.IGNORECASE),
        # "Kontakt\n\nName1, Name2, ..." — strip the whole Kontakt block
        re.compile(r"Kontakt\s*\n[^\n]*\n", re.IGNORECASE),
        # Phone numbers
        re.compile(r"Tlf\.?\s*:?\s*[\d\s]{8,}", re.IGNORECASE),
        re.compile(r"\b\d{2}\s\d{2}\s\d{2}\s\d{2}\b"),
    ]

    # Characters that count as "readable" for the garbled-text check
    _READABLE_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                "æøåÆØÅéèêëàáâüúùûöóòôïíìîäãñ "
                                ".,;:!?-–—\"'()\n\t")

    def _is_garbled(self, text: str, min_readable_ratio: float = 0.72) -> bool:
        """Return True if the text appears to be corrupted OCR or encoding garbage."""
        if not text:
            return True
        readable = sum(1 for c in text if c in self._READABLE_CHARS)
        return (readable / len(text)) < min_readable_ratio

    def _clean_text(self, text: str, source_type: DataSource) -> str:
        """Apply source-specific artifact removal."""
        if source_type == DataSource.DANMARKS_STATISTIK:
            for pat in self._DS_ARTIFACT_PATTERNS:
                text = pat.sub("", text)
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text

    def _adaptive_num_pairs(self, text: str) -> int:
        """Scale the number of requested pairs with article length."""
        n = len(text)
        if n < 500:
            return 1
        if n < 1500:
            return 2
        return self.config.num_pairs_per_article

    def create_prompt(self, article: Dict[str, Any], source_type: DataSource) -> str:
        """Build a source-aware prompt from the article's fields.

        The source_context is structured as two clearly labelled sections:
          ARTICLE METADATA  — labels the LLM uses for context / enriching questions
          ARTICLE TEXT      — the bare article body; this is what gets pasted into
                              text-grounded instructions
        """
        raw_text  = article.get("text", "")
        cleaned   = self._clean_text(raw_text, source_type)
        num_pairs = self._adaptive_num_pairs(cleaned)

        # Truncate body text; add ellipsis so the model knows it's cut
        if len(cleaned) > self.config.max_source_chars:
            body = cleaned[: self.config.max_source_chars] + "…"
        else:
            body = cleaned

        # ── Per-source metadata ──────────────────────────────────────────────
        if source_type == DataSource.WIKIPEDIA:
            meta = [f"Title: {article.get('title', '')}"]

        elif source_type == DataSource.DANMARKS_STATISTIK:
            meta = []
            if article.get("date"):         meta.append(f"Published: {article['date']}")
            if article.get("content_type"): meta.append(f"Type: {article['content_type']}")
            if article.get("series"):       meta.append(f"Series: {article['series']}")
            if article.get("title"):        meta.append(f"Title: {article['title']}")

        elif source_type == DataSource.DYNAWORD:
            meta = []
            if article.get("source"): meta.append(f"Source: {article['source']}")
            if article.get("date"):   meta.append(f"Date: {article['date']}")

        elif source_type == DataSource.TIDSSKRIFT:
            meta = []
            if article.get("journal"): meta.append(f"Journal: {article['journal']}")
            if article.get("date"):    meta.append(f"Published: {article['date']}")
            if article.get("authors"): meta.append(f"Authors: {article['authors']}")
            if article.get("title"):   meta.append(f"Title: {article['title']}")

        else:
            raise ValueError(f"Unsupported source type: {source_type}")

        # ── Assemble two-section context ─────────────────────────────────────
        source_context = (
            "ARTICLE METADATA (use as context; do NOT paste into instructions):\n"
            + "\n".join(meta)
            + "\n\nARTICLE TEXT (paste this verbatim for text-grounded instructions):\n"
            + body
        )

        return INSTRUCTION_PROMPT_TEMPLATE.format(
            num_pairs=num_pairs,
            source_context=source_context,
        )

    # ── API call ─────────────────────────────────────────────────────────────

    def clean_response(self, response: str) -> str:
        """Remove <think> blocks and markdown fences from LLM output."""
        cleaned = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE).sub("", response).strip()
        cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$",     "", cleaned, flags=re.MULTILINE)
        return cleaned

    def call_api_with_retry(self, article: Dict[str, Any], source_type: DataSource) -> Optional[str]:
        if not self.client:
            self.logger.warning("No API client — skipping")
            return None

        prompt = self.create_prompt(article, source_type)

        for attempt in range(self.config.max_retries):
            try:
                self.logger.debug(f"API attempt {attempt + 1}/{self.config.max_retries}")
                completion = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.config.max_tokens,
                    temperature=0.7,
                    stream=False,
                )
                raw = completion.choices[0].message.content or ""
                cleaned = self.clean_response(raw)
                self.logger.debug(f"API call successful, response length: {len(cleaned)}")
                return cleaned
            except Exception as e:
                self.logger.warning(f"API attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (2 ** attempt))
                else:
                    self.logger.error("All API attempts failed")
                    return None

    # ── JSON extraction ───────────────────────────────────────────────────────

    def extract_json_pairs(self, text: str) -> List[Dict[str, str]]:
        if not text:
            return []
        match = re.search(r"[\[\{].*[\]\}]", text, re.DOTALL)
        if not match:
            return []
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return self._validate_pairs(data)
            if isinstance(data, dict):
                if "instruction" in data and "response" in data:
                    return self._validate_pairs([data])
                for v in data.values():
                    if isinstance(v, list):
                        return self._validate_pairs(v)
        except json.JSONDecodeError:
            pass
        return []

    def _validate_pairs(self, pairs: List[Dict]) -> List[Dict[str, str]]:
        valid = []
        for pair in pairs:
            if not isinstance(pair, dict):
                continue
            if "responsive" in pair and "response" not in pair:
                pair["response"] = pair.pop("responsive")
            if "instruction" in pair and "response" in pair:
                pair["instruction"] = pair["instruction"].strip()
                pair["response"]    = pair["response"].strip()
                if pair["instruction"] and pair["response"]:
                    valid.append(pair)
        return valid


# ==================== Source Processor ====================

class SourceProcessor:
    """Handles generation for one data source with checkpointing."""

    def __init__(
        self,
        source_config: SourceConfig,
        generator: DanishDataGenerator,
        logger: logging.Logger,
        max_samples: Optional[int] = None,
    ):
        self.source_config   = source_config
        self.generator       = generator
        self.logger          = logger
        self.max_samples     = max_samples
        self.generation_state = self._load_generation_state()
        self.processed_ids   = set()
        self.statistics = {
            "total_processed": 0, "successful": 0, "failed": 0,
            "api_errors": 0, "parsing_errors": 0,
            "session_processed": 0, "cumulative_processed": 0,
        }

    def _load_generation_state(self) -> Dict[str, Any]:
        if os.path.exists(self.source_config.generation_state_file):
            try:
                with open(self.source_config.generation_state_file) as f:
                    state = json.load(f)
                    self.logger.info(
                        f"[{self.source_config.source_type.value}] Resumed: "
                        f"{state['total_articles_processed']} articles already done"
                    )
                    return state
            except Exception as e:
                self.logger.warning(f"Could not load state: {e}")
        return self._fresh_state()

    def _fresh_state(self) -> Dict[str, Any]:
        return {
            "source_type": self.source_config.source_type.value,
            "total_articles_processed": 0,
            "last_processed_index": -1,
            "last_processed_id": None,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "sessions": [],
        }

    def _save_generation_state(self):
        self.generation_state["last_updated"] = datetime.now().isoformat()
        with open(self.source_config.generation_state_file, "w") as f:
            json.dump(self.generation_state, f, indent=2)

    def get_article_id(self, article: Dict[str, Any]) -> str:
        src = self.source_config.source_type
        if src == DataSource.WIKIPEDIA:
            return str(article.get("url", "unknown"))
        if src == DataSource.DANMARKS_STATISTIK:
            return str(article.get("url", "unknown"))
        if src == DataSource.DYNAWORD:
            return str(article.get("id", "unknown"))
        if src == DataSource.TIDSSKRIFT:
            return str(article.get("doi") or article.get("url", "unknown"))
        return "unknown"

    def get_article_title(self, article: Dict[str, Any]) -> str:
        if self.source_config.source_type == DataSource.DYNAWORD:
            return article.get("text", "")[:80]
        return article.get("title", "")

    def load_checkpoint(self) -> Tuple[set, List[str]]:
        processed_ids = set()
        responses = []
        if os.path.exists(self.source_config.responses_output_file):
            self.logger.info(f"Loading checkpoint from {self.source_config.responses_output_file}")
            with open(self.source_config.responses_output_file, encoding="utf-8") as f:
                for line in f:
                    try:
                        result = json.loads(line)
                        processed_ids.add(str(result["source_id"]))
                        responses.append(result["synthetic_text"])
                    except (json.JSONDecodeError, KeyError):
                        pass
            self.logger.info(f"Resumed: {len(processed_ids)} articles already processed")
        return processed_ids, responses

    def generate_synthetic_data(self, source_data: List[Dict]) -> List[str]:
        processed_ids, model_responses = self.load_checkpoint()
        self.processed_ids = processed_ids

        start_index = self.generation_state["last_processed_index"] + 1

        if self.max_samples is not None:
            already = self.generation_state["total_articles_processed"]
            remaining = self.max_samples - already
            if remaining <= 0:
                self.logger.info(
                    f"[{self.source_config.source_type.value}] Already at limit "
                    f"({already}/{self.max_samples})"
                )
                return model_responses
            articles = source_data[start_index : start_index + remaining]
        else:
            articles = source_data[start_index:]

        if not articles:
            self.logger.info(f"[{self.source_config.source_type.value}] No new articles to process")
            return model_responses

        self.logger.info(
            f"[{self.source_config.source_type.value}] Processing {len(articles)} articles "
            f"(starting at index {start_index})"
        )

        session_info = {
            "start_time": datetime.now().isoformat(),
            "start_index": start_index,
            "planned_count": len(articles),
        }

        with open(self.source_config.responses_output_file, "a", encoding="utf-8") as f:
            for idx, article in enumerate(tqdm(articles, desc=f"Generating {self.source_config.source_type.value}")):
                article_id    = self.get_article_id(article)
                global_index  = start_index + idx

                if article_id in self.processed_ids:
                    continue

                if _is_boilerplate(article.get("text", "")):
                    self.logger.info(f"Skipping boilerplate article ID: {article_id}")
                    continue

                if self.generator._is_garbled(article.get("text", "")):
                    self.logger.info(f"Skipping garbled article ID: {article_id}")
                    continue

                self.logger.info(f"Processing ID: {article_id} (index: {global_index})")
                self.statistics["total_processed"]   += 1
                self.statistics["session_processed"] += 1

                synthetic = self.generator.call_api_with_retry(article, self.source_config.source_type)

                if synthetic:
                    model_responses.append(synthetic)
                    self.statistics["successful"] += 1

                    result = {
                        "source_id":    article_id,
                        "source_title": self.get_article_title(article),
                        "source_type":  self.source_config.source_type.value,
                        "synthetic_text": synthetic,
                        "global_index": global_index,
                        "timestamp":    time.time(),
                    }
                    json.dump(result, f, ensure_ascii=False)
                    f.write("\n")
                    f.flush()

                    self.processed_ids.add(article_id)
                    self.generation_state["total_articles_processed"] += 1
                    self.generation_state["last_processed_index"] = global_index
                    self.generation_state["last_processed_id"]    = article_id

                    if self.statistics["session_processed"] % 5 == 0:
                        self._save_generation_state()
                else:
                    self.statistics["failed"]     += 1
                    self.statistics["api_errors"] += 1

                time.sleep(self.generator.config.rate_limit_delay)

                if self.max_samples and self.generation_state["total_articles_processed"] >= self.max_samples:
                    self.logger.info(f"[{self.source_config.source_type.value}] Reached limit")
                    break

        session_info["end_time"]          = datetime.now().isoformat()
        session_info["articles_processed"] = self.statistics["session_processed"]
        session_info["end_index"]          = self.generation_state["last_processed_index"]
        self.generation_state["sessions"].append(session_info)
        self._save_generation_state()

        self.statistics["cumulative_processed"] = self.generation_state["total_articles_processed"]
        return model_responses

    # Instructions longer than this threshold are assumed to paste the article text
    _GROUNDED_THRESHOLD = 400

    def _limit_grounded_pairs(self, pairs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Keep at most 1 text-grounded pair per article response; keep all standalone pairs."""
        grounded_seen = 0
        filtered = []
        for pair in pairs:
            if len(pair.get("instruction", "")) > self._GROUNDED_THRESHOLD:
                if grounded_seen == 0:
                    filtered.append(pair)
                    grounded_seen += 1
                # else: drop extra grounded pairs silently
            else:
                filtered.append(pair)
        return filtered

    def process_and_save_results(self, model_responses: List[str]) -> pd.DataFrame:
        self.logger.info(f"[{self.source_config.source_type.value}] Processing responses…")

        combined_data, empty_count, dropped_grounded = [], 0, 0
        for response in model_responses:
            if not response:
                empty_count += 1
                continue
            pairs = self.generator.extract_json_pairs(response)
            if pairs:
                before = len(pairs)
                pairs = self._limit_grounded_pairs(pairs)
                dropped_grounded += before - len(pairs)
                for pair in pairs:
                    pair["source_type"] = self.source_config.source_type.value
                combined_data.extend(pairs)
            else:
                self.statistics["parsing_errors"] += 1

        self.logger.info(
            f"[{self.source_config.source_type.value}] Extracted {len(combined_data)} pairs "
            f"({empty_count} empty, {dropped_grounded} duplicate-grounded dropped)"
        )

        # Add ShareGPT conversation structure
        for item in combined_data:
            item["conversations"] = [
                {"from": "human", "value": item.get("instruction", "")},
                {"from": "gpt",   "value": item.get("response",    "")},
            ]

        with open(self.source_config.final_json_output, "w", encoding="utf-8") as f:
            json.dump(combined_data, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Saved JSON to {self.source_config.final_json_output}")

        df = pd.DataFrame(combined_data)
        if not df.empty:
            df.to_csv(self.source_config.final_csv_output, index=False, encoding="utf-8")
            self.logger.info(f"Saved CSV to {self.source_config.final_csv_output}")

        return df


# ==================== Pipeline Orchestration ====================

def _source_cfg_and_limit(source_type: DataSource, config: Config):
    """Return (source_config, max_samples) for the given source."""
    if source_type == DataSource.WIKIPEDIA:
        return config.wikipedia_config, (config.max_wikipedia_samples or config.max_samples_per_source)
    if source_type == DataSource.DANMARKS_STATISTIK:
        return config.danmarks_statistik_config, (config.max_danmarks_statistik_samples or config.max_samples_per_source)
    if source_type == DataSource.DYNAWORD:
        return config.dynaword_config, (config.max_dynaword_samples or config.max_samples_per_source)
    if source_type == DataSource.TIDSSKRIFT:
        return config.tidsskrift_config, (config.max_tidsskrift_samples or config.max_samples_per_source)
    raise ValueError(f"Unknown source: {source_type}")


def process_source(
    source_type: DataSource,
    config: Config,
    logger: logging.Logger,
    generator: DanishDataGenerator,
) -> Optional[pd.DataFrame]:

    source_config, max_samples = _source_cfg_and_limit(source_type, config)

    if not os.path.exists(source_config.input_file):
        logger.warning(f"Input file not found: {source_config.input_file} — skipping")
        return None

    logger.info(f"Loading {source_type.value} data from {source_config.input_file}")
    with open(source_config.input_file, encoding="utf-8") as f:
        source_data = json.load(f)
    logger.info(f"Loaded {len(source_data):,} articles")

    processor    = SourceProcessor(source_config, generator, logger, max_samples)
    already_done = processor.generation_state["total_articles_processed"]
    if already_done > 0:
        logger.info(f"[{source_type.value}] Previous progress: {already_done} articles processed")

    model_responses = processor.generate_synthetic_data(source_data)

    if processor.statistics["session_processed"] > 0:
        df = processor.process_and_save_results(model_responses)
        logger.info(f"[{source_type.value}] session={processor.statistics['session_processed']} "
                    f"ok={processor.statistics['successful']} "
                    f"fail={processor.statistics['failed']} "
                    f"total={processor.statistics['cumulative_processed']}")
        return df
    else:
        logger.info(f"[{source_type.value}] No new articles processed")
        if os.path.exists(source_config.final_csv_output):
            return pd.read_csv(source_config.final_csv_output)
        return None


def combine_datasets(
    datasets: Dict[DataSource, pd.DataFrame],
    config: Config,
    logger: logging.Logger,
) -> pd.DataFrame:
    if not datasets:
        logger.warning("No datasets to combine")
        return pd.DataFrame()

    combined_df = pd.concat(datasets.values(), ignore_index=True)
    logger.info(f"Saving combined dataset with {len(combined_df):,} pairs")

    with open(config.combined_output_json, "w", encoding="utf-8") as f:
        json.dump(combined_df.to_dict("records"), f, indent=2, ensure_ascii=False)
    logger.info(f"Saved combined JSON to {config.combined_output_json}")

    combined_df.to_csv(config.combined_output_csv, index=False, encoding="utf-8")
    logger.info(f"Saved combined CSV to {config.combined_output_csv}")

    return combined_df


def combine_existing_datasets(config: Optional[Config] = None):
    """Combine already-generated per-source output files."""
    if config is None:
        config = Config()
    logger = setup_logging(config.log_file)
    logger.info("Combining existing datasets…")

    source_map = {
        DataSource.WIKIPEDIA:          config.wikipedia_config,
        DataSource.DANMARKS_STATISTIK: config.danmarks_statistik_config,
        DataSource.DYNAWORD:           config.dynaword_config,
        DataSource.TIDSSKRIFT:         config.tidsskrift_config,
    }

    dfs = {}
    for src, src_cfg in source_map.items():
        if os.path.exists(src_cfg.final_csv_output):
            df = pd.read_csv(src_cfg.final_csv_output)
            dfs[src] = df
            logger.info(f"Loaded {src.value}: {len(df):,} pairs")

    if len(dfs) < 2:
        logger.warning("Need at least 2 source files to combine")
        return None

    combined_df = combine_datasets(dfs, config, logger)
    print(f"Combined {len(combined_df):,} total pairs → {config.combined_output_json}")
    return combined_df


# ==================== Main ====================

def main(
    sources: Optional[List[DataSource]] = None,
    model_name: Optional[str] = None,
    max_samples_per_source:         Optional[int] = None,
    max_wikipedia_samples:          Optional[int] = None,
    max_danmarks_statistik_samples: Optional[int] = None,
    max_dynaword_samples:           Optional[int] = None,
    max_tidsskrift_samples:         Optional[int] = None,
):
    config = Config()

    if model_name                     is not None:
        config.model_name                     = model_name
    if max_samples_per_source         is not None:
        config.max_samples_per_source         = max_samples_per_source
    if max_wikipedia_samples          is not None:
        config.max_wikipedia_samples          = max_wikipedia_samples
    if max_danmarks_statistik_samples is not None:
        config.max_danmarks_statistik_samples = max_danmarks_statistik_samples
    if max_dynaword_samples           is not None:
        config.max_dynaword_samples           = max_dynaword_samples
    if max_tidsskrift_samples         is not None:
        config.max_tidsskrift_samples         = max_tidsskrift_samples

    if sources is None:
        sources = list(DataSource)

    logger = setup_logging(config.log_file)
    logger.info("=" * 60)
    logger.info("Starting Danish Synthetic Data Generation")
    logger.info(f"Model:   {config.model_name}")
    logger.info(f"Sources: {[s.value for s in sources]}")
    logger.info("=" * 60)

    try:
        generator = DanishDataGenerator(config, logger)
        datasets  = {}

        for source_type in sources:
            logger.info(f"\nProcessing {source_type.value}…")
            df = process_source(source_type, config, logger, generator)
            if df is not None and not df.empty:
                datasets[source_type] = df
                logger.info(f"✓ {source_type.value}: {len(df):,} instruction-response pairs")

        if len(datasets) > 1:
            combine_datasets(datasets, config, logger)
        elif not datasets:
            logger.warning("No data generated from any source")

        total = sum(len(df) for df in datasets.values())
        logger.info(f"\nTotal instruction-response pairs generated: {total:,}")

        print("\n✅ Done!")
        print(f"📊 Generated {total:,} total instruction-response pairs")
        print("\n📁 Output files:")
        for src_type, src_cfg in [
            (DataSource.WIKIPEDIA,          config.wikipedia_config),
            (DataSource.DANMARKS_STATISTIK, config.danmarks_statistik_config),
            (DataSource.DYNAWORD,           config.dynaword_config),
            (DataSource.TIDSSKRIFT,         config.tidsskrift_config),
        ]:
            if src_type in datasets and os.path.exists(src_cfg.final_json_output):
                print(f"  {src_type.value}: {src_cfg.final_json_output}")
        if len(datasets) > 1:
            print(f"  Combined: {config.combined_output_json}")
        print(f"  Log: {config.log_file}")

    except Exception as e:
        logger = logging.getLogger("DaDataGenerator")
        logger.error(f"Unexpected error: {e}\n{traceback.format_exc()}")
        raise


# ==================== CLI ====================

if __name__ == "__main__":
    args = sys.argv[1:]

    if "--help" in args or "-h" in args:
        print(__doc__)
        sys.exit(0)

    if "--combine-only" in args:
        combine_existing_datasets()
        sys.exit(0)

    # ── Source selection ─────────────────────────────────────────────────────
    source_map = {
        "wikipedia":          DataSource.WIKIPEDIA,
        "danmarks_statistik": DataSource.DANMARKS_STATISTIK,
        "dynaword":           DataSource.DYNAWORD,
        "tidsskrift":         DataSource.TIDSSKRIFT,
    }

    sources = None
    if "--wikipedia-only" in args:
        sources = [DataSource.WIKIPEDIA]
    elif "--danmarks-statistik-only" in args:
        sources = [DataSource.DANMARKS_STATISTIK]
    elif "--dynaword-only" in args:
        sources = [DataSource.DYNAWORD]
    elif "--tidsskrift-only" in args:
        sources = [DataSource.TIDSSKRIFT]
    elif "--sources" in args:
        idx = args.index("--sources")
        names = args[idx + 1].split(",")
        sources = [source_map[n.strip()] for n in names if n.strip() in source_map]

    # ── Sample limits ────────────────────────────────────────────────────────
    def _int_arg(flag):
        if flag in args:
            i = args.index(flag)
            try:
                return int(args[i + 1])
            except (IndexError, ValueError):
                print(f"Error: {flag} must be followed by an integer")
                sys.exit(1)
        return None

    def _str_arg(flag):
        if flag in args:
            i = args.index(flag)
            if i + 1 < len(args):
                return args[i + 1]
        return None

    main(
        sources=sources,
        model_name                     = _str_arg("--model"),
        max_samples_per_source         = _int_arg("--max-samples"),
        max_wikipedia_samples          = _int_arg("--max-wikipedia"),
        max_danmarks_statistik_samples = _int_arg("--max-danmarks-statistik"),
        max_dynaword_samples           = _int_arg("--max-dynaword"),
        max_tidsskrift_samples         = _int_arg("--max-tidsskrift"),
    )
