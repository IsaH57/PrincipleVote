#!/usr/bin/env python3
"""Run multiple image reward models on HPDv2 rankings and compare against human preferences.

This script loads the HPDv2 prompt/image rankings, converts the human annotations
into ``pref_voting`` profiles, scores every prompt with a configurable set of
reward models, and computes ranking correlation metrics against the human
consensus (Borda) ranking.

Example usage::

    python run_reward_model_benchmark.py \
        --dataset /home/ra63hik/vision_icai/datasets/hpdv2_test_export/hpdv2_test_rankings.json \
        --image-root /home/ra63hik/vision_icai/datasets/hpdv2_test_export/images \
        --output ranking_results/model_reward_benchmark.json \
    --models hpsv2 open_clip image_reward \
    --hpsv2-version v2.1 \
        --openclip-pretrained laion2b_s32b_b79k

By default only the OpenCLIP baseline is enabled since it does not require any
fine-tuned checkpoints. Other models can be toggled through the CLI and will be
skipped gracefully if their dependencies are unavailable.

The ``--dataset`` flag accepts either a rankings file or a directory containing
standard HPDv2 exports (e.g. ``hpdv2_test_rankings.json``).
"""
from __future__ import annotations
#this is clearly wrong at the moment. crafully go thorugh the entire principlevote repository and adjust the reward model benchmark so that it works with the standards set in the repository. the goal is simply to have the different reward models benchmarked in the aem way as already done in the base repository for hpvs2. the data folder structure might be different. boith strucutres ours and the repo expepcted should be supported without making it spaghetti code. our data can be found at /home/ra63hik/vision_icai/datasets/hpdv2_test_export.
import argparse
import csv
import json
import logging
import math
import os
import shutil
import sys
import types
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from tqdm import tqdm

from pref_voting.profiles import Profile
from pref_voting.scoring_methods import borda, borda_ranking, plurality
from pref_voting.c1_methods import copeland  # type: ignore[import]
from pref_voting.rankings import Ranking

# Third-party model imports are intentionally deferred to their wrapper classes so
# optional dependencies do not become hard requirements for running the script.

SUMMARY_METRICS = [
    "kendall_tau",
    "spearman_rho",
    "top1_match",
    "top3_overlap",
    "condorcet_top1",
    "exact_position_match",
    "winner_agreement_borda",
    "winner_agreement_plurality",
    "winner_agreement_copeland",
]


@dataclass
class PromptRecord:
    """Canonical representation of a single HPDv2 prompt entry."""

    prompt_index: Optional[int]
    prompt: str
    image_paths: List[str]
    image_filenames: List[str]
    rankings: List[List[int]]
    counts: List[int]
    raw_entry: Dict[str, Any]


class RewardModel:
    """Base class for all reward models used in the benchmark."""

    def __init__(self, name: str) -> None:
        self.name = name

    def score(self, prompt: str, image_paths: Sequence[str]) -> List[float]:  # pragma: no cover - runtime only
        raise NotImplementedError

    def rank(self, prompt: str, image_paths: Sequence[str]) -> Tuple[List[int], List[float]]:
        """Return candidate indices sorted by descending score along with raw scores."""

        scores = self.score(prompt, image_paths)
        order = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
        return order, [float(s) for s in scores]


def bootstrap_hpsv2(explicit_root: Optional[str] = None) -> None:
    """Ensure the HPSv2 repository is available on sys.path."""

    if getattr(bootstrap_hpsv2, "_bootstrapped", False):
        return

    candidate_paths: List[Path] = []
    if explicit_root:
        candidate_paths.append(Path(explicit_root))

    env_root = os.environ.get("HPSV2_ROOT")
    if env_root:
        candidate_paths.append(Path(env_root))

    script_dir = Path(__file__).resolve().parent
    candidate_paths.extend(
        [
            script_dir,
            script_dir.parent,
            script_dir.parent.parent,
        ]
    )

    normalized: List[Path] = []
    for cand in candidate_paths:
        cand = cand.resolve()
        if not cand.exists():
            continue
        normalized.append(cand)
        for alt in (cand / "HPSv2", cand / "hpsv2"):
            if alt.exists():
                normalized.append(alt.resolve())

    for cand in normalized:
        package_root = cand if (cand / "hpsv2").is_dir() else None
        if package_root is None:
            continue
        if str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))
        bootstrap_hpsv2._bootstrapped = True
        return

    logging.warning(
        "HPSv2 repository not found automatically. Set the HPSV2_ROOT environment variable if local sources are required."
    )


class OpenCLIPRewardModel(RewardModel):
    """Vanilla OpenCLIP similarity scoring."""

    def __init__(
        self,
        model_name: str,
        pretrained: str,
        device: torch.device,
        precision: str,
    ) -> None:
        super().__init__(name=f"open_clip::{model_name}::{pretrained}")

        try:
            from open_clip import create_model_and_transforms, get_tokenizer  # type: ignore
        except ImportError:  # pragma: no cover - fallback path for legacy setup
            bootstrap_hpsv2()
            from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer  # type: ignore

        self.device = device
        self.use_autocast = device.type == "cuda" and precision == "amp"

        self.model, _, self.preprocess = create_model_and_transforms(
            model_name,
            pretrained,
            precision=precision,
            device=device,
            jit=False,
            output_dict=True,
        )
        self.model.eval()
        self.tokenizer = get_tokenizer(model_name)

    def score(self, prompt: str, image_paths: Sequence[str]) -> List[float]:
        images: List[torch.Tensor] = []
        for img_path in image_paths:
            with Image.open(img_path).convert("RGB") as img:
                images.append(self.preprocess(img))

        image_batch = torch.stack(images).to(self.device)
        text_tokens = self.tokenizer([prompt] * len(images)).to(self.device)

        autocast_ctx = torch.cuda.amp.autocast if self.use_autocast else nullcontext

        with torch.no_grad():
            with autocast_ctx():
                outputs = self.model(image_batch, text_tokens)
                image_features = outputs["image_features"]
                text_features = outputs["text_features"]
                logit_scale = outputs["logit_scale"]
                logits_per_image = logit_scale * image_features @ text_features.T
                scores = torch.diagonal(logits_per_image).float().cpu().tolist()

        return scores


class HPSv2RewardModel(RewardModel):
    """HPSv2 reward via the official PyPI package."""

    def __init__(self, version: str, device: torch.device) -> None:
        try:
            import turtle  # noqa: F401
        except ModuleNotFoundError:
            stub = types.ModuleType("turtle")

            def _noop_forward(*_: object, **__: object) -> None:
                return None

            stub.forward = _noop_forward  # type: ignore[attr-defined]
            sys.modules.setdefault("turtle", stub)

        try:
            import hpsv2  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Package 'hpsv2' is required for the HPSv2 reward model. Run `pip install hpsv2`."
            ) from exc

        super().__init__(name=f"hpsv2::{version}")
        self.version = version
        self.device = device
        self._hpsv2 = hpsv2
        self._ensure_tokenizer_vocab()

    def score(self, prompt: str, image_paths: Sequence[str]) -> List[float]:
        if not image_paths:
            return []

        # The library expects plain Python lists.
        images_list = list(image_paths)
        result = self._hpsv2.score(images_list, prompt, hps_version=self.version)

        # Normalize various possible return formats into a list of floats following image_paths order.
        raw_scores: object = result
        if isinstance(result, dict):
            if "scores" in result:
                raw_scores = result["scores"]
            elif "score" in result:
                raw_scores = result["score"]

        if hasattr(raw_scores, "tolist"):
            raw_scores = raw_scores.tolist()  # type: ignore[assignment]

        scores: List[float] = []
        if isinstance(raw_scores, dict):
            for img_path in images_list:
                score_val = raw_scores.get(img_path)
                if score_val is None:
                    score_val = raw_scores.get(os.path.basename(img_path))
                if score_val is None:
                    raise ValueError(
                        f"HPSv2 did not return a score for image '{img_path}'."
                    )
                scores.append(float(score_val))
        elif isinstance(raw_scores, (list, tuple)):
            scores = [float(s) for s in raw_scores]
        else:
            try:
                scores = [float(s) for s in list(raw_scores)]  # type: ignore[arg-type]
            except TypeError as exc:  # pragma: no cover - defensive guard
                raise ValueError(
                    f"Unexpected HPSv2 score format: {type(raw_scores)!r}"
                ) from exc

        if len(scores) != len(images_list):
            raise ValueError(
                "HPSv2 returned a different number of scores than images provided."
            )

        return scores

    def _ensure_tokenizer_vocab(self) -> None:
        """Copy the CLIP tokenizer vocab if the package did not ship with one."""

        try:
            base_dir = Path(self._hpsv2.__file__).resolve().parent
        except Exception:  # pragma: no cover - defensive guard
            return

        target = base_dir / "src" / "open_clip" / "bpe_simple_vocab_16e6.txt.gz"
        if target.exists():
            return

        try:
            import open_clip  # type: ignore
        except ImportError:
            logging.warning(
                "Unable to find tokenizer vocab for HPSv2; install `open-clip-torch` or provide the file at %s",
                target,
            )
            return

        source = Path(open_clip.__file__).resolve().parent / "bpe_simple_vocab_16e6.txt.gz"
        if not source.exists():
            logging.warning(
                "open_clip package does not contain tokenizer vocab at %s", source
            )
            return

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source, target)
            logging.info("Copied tokenizer vocab from %s to %s", source, target)
        except OSError as exc:  # pragma: no cover - IO guard
            logging.warning("Failed to copy tokenizer vocab for HPSv2: %s", exc)


class ImageRewardWrapper(RewardModel):
    """Wrapper for the ImageReward model (optional dependency)."""

    def __init__(self, model_name: str, device: torch.device, cache_dir: Optional[str] = None) -> None:
        try:
            import ImageReward as RM  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                "ImageReward is not installed. Please `pip install image-reward` to enable this model."
            ) from exc

        super().__init__(name=f"image_reward::{model_name}")
        self.device = device
        # ImageReward.load expects a device string; allow cache override via download_root
        download_root = cache_dir or os.path.expanduser("~/.cache/ImageReward")
        self.model = RM.load(model_name, device=str(device), download_root=download_root)

    def score(self, prompt: str, image_paths: Sequence[str]) -> List[float]:
        # ImageReward.score accepts either a single path or a list; we batch for efficiency
        if not image_paths:
            return []

        rewards = self.model.score(prompt, list(image_paths))
        # Some versions return nested lists, ensure flat float list in original order
        flattened: List[float] = []
        for item in rewards:
            if isinstance(item, (list, tuple)) and len(item) == 1:
                flattened.append(float(item[0]))
            else:
                flattened.append(float(item))
        return flattened


def _read_dataset_entries(dataset_path: Path) -> List[Dict[str, Any]]:
    """Load dataset rows from either JSON or JSONL input."""

    if dataset_path.suffix.lower() == ".jsonl":
        entries: List[Dict[str, Any]] = []
        with dataset_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                entries.append(json.loads(line))
        return entries

    with dataset_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in ("data", "prompts", "entries", "items", "records"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                return candidate

    raise ValueError(f"Unsupported dataset structure in {dataset_path}")


def _resolve_image_path(raw_path: str, dataset_dir: Path, image_root: Optional[Path]) -> str:
    """Best-effort resolution for image locations across varying folder layouts."""

    if not raw_path:
        raise ValueError("Encountered empty image path entry.")

    raw_path = raw_path.strip()
    # Common dataset exports use Windows-style separators.
    normalized_raw = raw_path.replace("\\", "/")
    candidate_paths: List[Path] = []

    raw_candidate = Path(normalized_raw)
    if raw_candidate.is_absolute():
        candidate_paths.append(raw_candidate)
    else:
        if image_root is not None:
            candidate_paths.append(image_root / raw_candidate)
        candidate_paths.append(dataset_dir / raw_candidate)
        if image_root is not None:
            candidate_paths.append(image_root / raw_candidate.name)
        candidate_paths.append(dataset_dir / raw_candidate.name)

    seen: set[str] = set()
    ordered_candidates: List[Path] = []
    for cand in candidate_paths:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        ordered_candidates.append(cand)

    for cand in ordered_candidates:
        if cand.exists():
            try:
                return str(cand.resolve(strict=False))
            except OSError:
                return str(cand)

    fallback = ordered_candidates[0] if ordered_candidates else raw_candidate
    missing_key = str(fallback)
    warned: set[str] = getattr(_resolve_image_path, "_warned", set())
    if missing_key not in warned:
        logging.warning(
            "Could not verify image path '%s'. Using best-effort path '%s'. Provide --image-root if needed.",
            raw_path,
            fallback,
        )
        warned.add(missing_key)
        setattr(_resolve_image_path, "_warned", warned)

    return str(fallback)


def _normalize_ranking(
    ranking: Sequence[Any],
    num_candidates: int,
    assume_positions: bool = False,
    expected_top: Optional[int] = None,
) -> List[int]:
    """Normalize a sequence representing candidate order into a full permutation."""

    cleaned: List[int] = []
    seen: set[int] = set()
    for item in ranking:
        try:
            idx = int(item)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < num_candidates and idx not in seen:
            cleaned.append(idx)
            seen.add(idx)

    for idx in range(num_candidates):
        if idx not in seen:
            cleaned.append(idx)

    return cleaned[:num_candidates]


def _positions_to_order(values: Sequence[Any], num_candidates: int) -> List[int]:
    """Convert an array of per-candidate rank positions into an ordered list."""

    ranked_pairs: List[Tuple[int, int]] = []
    for candidate_idx, raw in enumerate(values):
        if candidate_idx >= num_candidates:
            break
        try:
            rank_val = int(raw)
        except (TypeError, ValueError):
            continue
        ranked_pairs.append((rank_val, candidate_idx))

    ranked_pairs.sort(key=lambda pair: (pair[0], pair[1]))
    order = [candidate for _, candidate in ranked_pairs]
    if len(order) < num_candidates:
        missing = [idx for idx in range(num_candidates) if idx not in order]
        order.extend(missing)
    return order[:num_candidates]


def _extract_image_metadata(
    entry: Dict[str, Any],
    dataset_dir: Path,
    image_root: Optional[Path],
) -> Tuple[List[str], List[str]]:
    """Return resolved image paths and filenames for a dataset entry."""

    raw_paths: List[str] = []
    filenames: List[str] = []

    images_field = entry.get("images")
    if isinstance(images_field, list) and images_field:
        for idx, image_item in enumerate(images_field):
            raw_path = None
            filename = None
            if isinstance(image_item, dict):
                raw_path = (
                    image_item.get("path")
                    or image_item.get("image_path")
                    or image_item.get("relative_path")
                    or image_item.get("filepath")
                    or image_item.get("file")
                )
                filename = image_item.get("filename")
            else:
                raw_path = str(image_item)

            if raw_path is None:
                raise ValueError(f"Image entry {idx} is missing a path attribute.")

            if not filename:
                filename = Path(raw_path).name or f"image_{idx}"

            raw_paths.append(str(raw_path))
            filenames.append(filename)

    elif isinstance(entry.get("image_path"), list):
        raw_paths = [str(p) for p in entry["image_path"]]
        filenames = [Path(p).name or f"image_{idx}" for idx, p in enumerate(raw_paths)]

    elif isinstance(entry.get("image_paths"), list):
        raw_paths = [str(p) for p in entry["image_paths"]]
        filenames = [Path(p).name or f"image_{idx}" for idx, p in enumerate(raw_paths)]

    else:
        raise ValueError("Unable to locate image paths for dataset entry.")

    resolved_paths = [
        _resolve_image_path(raw_path, dataset_dir, image_root)
        for raw_path in raw_paths
    ]

    return resolved_paths, filenames


def _extract_rankings(entry: Dict[str, Any], num_candidates: int) -> Tuple[List[List[int]], List[int]]:
    """Gather per-annotator rankings and associated counts."""

    rankings: List[List[int]] = []
    counts: List[int] = []

    position_like_keys = {"ranking_indices", "ranking", "rank", "annotation", "scores", "consensus"}

    def to_order(raw: Sequence[Any], key_hint: Optional[str] = None) -> List[int]:
        if key_hint is not None and key_hint in position_like_keys:
            return _positions_to_order(raw, num_candidates)
        return _normalize_ranking(raw, num_candidates)

    def append_ranking(raw_ranking: Sequence[Any], weight: int = 1, key_hint: Optional[str] = None) -> None:
        normalized = to_order(raw_ranking, key_hint)
        rankings.append(normalized)
        counts.append(max(int(weight), 1))

    annotators = entry.get("annotators")
    if isinstance(annotators, list):
        for annot in annotators:
            if not isinstance(annot, dict):
                continue
            if annot.get("ranking_indices") is not None:
                append_ranking(annot["ranking_indices"], key_hint="ranking_indices")
                continue
            if annot.get("order") is not None:
                append_ranking(annot["order"], key_hint="order")
                continue
            if annot.get("preference") is not None:
                append_ranking(annot["preference"], key_hint="preference")
                continue
            if annot.get("ranking") is not None:
                append_ranking(annot["ranking"], key_hint="ranking")
                continue

            annotation_scores = annot.get("annotation") or annot.get("scores")
            if annotation_scores is not None:
                order = _positions_to_order(annotation_scores, num_candidates)
                append_ranking(order)

    if not rankings and isinstance(entry.get("rankings"), list):
        ranking_list = entry["rankings"]
        ranking_counts = entry.get("ranking_counts")
        for idx, ranking in enumerate(ranking_list):
            weight = 1
            if isinstance(ranking_counts, list) and idx < len(ranking_counts):
                try:
                    weight = int(ranking_counts[idx])
                except (TypeError, ValueError):
                    weight = 1
            append_ranking(ranking, weight)

    if not rankings and entry.get("raw_annotations"):
        raw_annotations = entry["raw_annotations"]
        if isinstance(raw_annotations, list):
            for annot in raw_annotations:
                if not isinstance(annot, dict):
                    continue
                annotation_scores = annot.get("annotation") or annot.get("scores")
                if annotation_scores is None:
                    continue
                order = _positions_to_order(annotation_scores, num_candidates)
                append_ranking(order)

    if not rankings:
        if entry.get("ranking") is not None:
            append_ranking(entry["ranking"], key_hint="ranking")
        elif entry.get("rank") is not None:
            append_ranking(entry["rank"], key_hint="rank")

    if not rankings:
        raise ValueError("No rankings found for dataset entry.")

    return rankings, counts


def _coerce_single_winner(value: object) -> Optional[int]:
    """Extract a representative winner index from various pref_voting return types."""

    if value is None:
        return None

    if isinstance(value, (int, float)):
        return int(value)

    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        for item in value:
            try:
                return int(item)
            except (TypeError, ValueError):
                continue
        return None

    return None


def kendall_tau(order_a: Sequence[int], order_b: Sequence[int]) -> float:
    """Compute the Kendall tau rank correlation coefficient."""

    if len(order_a) <= 1:
        return 1.0

    index_b = {item: idx for idx, item in enumerate(order_b)}
    concordant = 0
    discordant = 0

    for i in range(len(order_a)):
        for j in range(i + 1, len(order_a)):
            a_i = order_a[i]
            a_j = order_a[j]
            if index_b[a_i] < index_b[a_j]:
                concordant += 1
            else:
                discordant += 1

    total_pairs = concordant + discordant
    if total_pairs == 0:
        return 1.0
    return (concordant - discordant) / total_pairs


def spearman_rho(order_a: Sequence[int], order_b: Sequence[int]) -> float:
    """Compute Spearman's rank correlation coefficient."""

    n = len(order_a)
    if n <= 1:
        return 1.0

    rank_a = {item: idx for idx, item in enumerate(order_a)}
    rank_b = {item: idx for idx, item in enumerate(order_b)}
    diff_sq_sum = sum((rank_a[item] - rank_b[item]) ** 2 for item in order_a)
    return 1.0 - (6.0 * diff_sq_sum) / (n * (n * n - 1))


def normalized_rank_agreement(order_a: Sequence[int], order_b: Sequence[int]) -> float:
    """Return fraction of positions where both rankings agree exactly."""

    matches = sum(int(a == b) for a, b in zip(order_a, order_b))
    return matches / len(order_a) if order_a else 1.0


def compute_metrics(
    human_order: Sequence[int],
    model_order: Sequence[int],
    condorcet_winner: Optional[int],
    borda_winners: Sequence[int],
    plurality_winners: Sequence[int],
    copeland_winners: Sequence[int],
) -> Dict[str, Optional[float]]:
    tau = kendall_tau(human_order, model_order)
    rho = spearman_rho(human_order, model_order)
    top1 = float(model_order[0] == human_order[0]) if model_order else None
    top3 = None
    if len(model_order) >= 3 and len(human_order) >= 3:
        top3 = float(len(set(model_order[:3]) & set(human_order[:3])) > 0)
    condorcet_hit = None
    if condorcet_winner is not None and model_order:
        condorcet_hit = float(model_order[0] == condorcet_winner)
    exact = normalized_rank_agreement(human_order, model_order)

    winner_match_borda = None
    winner_match_plurality = None
    winner_match_copeland = None
    if model_order:
        winner = model_order[0]
        if borda_winners:
            winner_match_borda = float(winner in set(borda_winners))
        if plurality_winners:
            winner_match_plurality = float(winner in set(plurality_winners))
        if copeland_winners:
            winner_match_copeland = float(winner in set(copeland_winners))

    return {
        "kendall_tau": tau,
        "spearman_rho": rho,
        "top1_match": top1,
        "top3_overlap": top3,
        "condorcet_top1": condorcet_hit,
        "exact_position_match": exact,
        "winner_agreement_borda": winner_match_borda,
        "winner_agreement_plurality": winner_match_plurality,
        "winner_agreement_copeland": winner_match_copeland,
    }


def consensus_order(profile: Profile) -> List[int]:
    """Return a deterministic linear order using Borda with alphabetical tie-breaking."""

    ranking: Ranking = borda_ranking(profile, tie_breaking="alphabetic")
    as_tuple = ranking.to_linear()
    if as_tuple is None:  # pragma: no cover - tie-breaking should make it linear, but guard anyway
        raise ValueError("Borda ranking still contains ties; cannot produce linear order.")
    return list(as_tuple)


def build_profile(record: PromptRecord) -> Profile:
    """Instantiate a pref_voting Profile from a normalized prompt record."""

    counts = record.counts if any(count != 1 for count in record.counts) else None
    profile = Profile(record.rankings, counts)
    profile.prompt = record.prompt
    profile.prompt_index = record.prompt_index  # type: ignore[attr-defined]
    profile.image_paths = record.image_paths
    profile.alternatives_name = {
        idx: name for idx, name in enumerate(record.image_filenames)
    }
    return profile


def load_dataset(
    dataset_path: Path,
    max_prompts: Optional[int] = None,
    image_root: Optional[Path] = None,
) -> List[PromptRecord]:
    """Load and normalize prompts across differing HPDv2 export formats."""

    if image_root is not None:
        image_root = image_root.resolve()

    if dataset_path.is_dir():
        for candidate_name in (
            "hpdv2_test_rankings.json",
            "test.json",
            "test_rankings.json",
            "rankings.json",
        ):
            candidate_path = dataset_path / candidate_name
            if candidate_path.exists():
                dataset_path = candidate_path
                break
        else:
            raise ValueError(
                f"Could not locate a dataset JSON file inside {dataset_path}."
            )

    entries = _read_dataset_entries(dataset_path)
    dataset_dir = dataset_path.parent

    normalized: List[PromptRecord] = []
    for idx, entry in enumerate(entries):
        prompt_text = entry.get("prompt") or entry.get("text") or entry.get("instruction")
        if prompt_text is None:
            prompt_text = ""
        prompt_text = str(prompt_text)

        prompt_index = entry.get("prompt_index")
        if not isinstance(prompt_index, int):
            alt_index = entry.get("index") or entry.get("id") or entry.get("prompt_id")
            if isinstance(alt_index, int):
                prompt_index = alt_index
            elif isinstance(alt_index, str) and alt_index.isdigit():
                prompt_index = int(alt_index)
            else:
                prompt_index = None

        try:
            image_paths, image_filenames = _extract_image_metadata(entry, dataset_dir, image_root)
            rankings, counts = _extract_rankings(entry, len(image_paths))
        except ValueError as exc:
            logging.warning("Skipping prompt %s: %s", prompt_index if prompt_index is not None else idx, exc)
            continue

        normalized.append(
            PromptRecord(
                prompt_index=prompt_index,
                prompt=prompt_text,
                image_paths=image_paths,
                image_filenames=image_filenames,
                rankings=rankings,
                counts=counts,
                raw_entry=entry,
            )
        )

        if max_prompts is not None and len(normalized) >= max_prompts:
            break

    if not normalized:
        raise ValueError("No valid prompts were found in the provided dataset.")

    return normalized


def prepare_models(
    args: argparse.Namespace,
    device: torch.device,
) -> List[RewardModel]:
    models: List[RewardModel] = []

    for model_name in args.models:
        key = model_name.lower()
        if key == "open_clip":
            models.append(
                OpenCLIPRewardModel(
                    model_name=args.openclip_model,
                    pretrained=args.openclip_pretrained,
                    device=device,
                    precision=args.precision,
                )
            )
        elif key == "hpsv2":
            try:
                models.append(
                    HPSv2RewardModel(
                        version=args.hpsv2_version,
                        device=device,
                    )
                )
            except RuntimeError as exc:
                logging.warning("Skipping HPSv2 model: %s", exc)
        elif key == "image_reward":
            try:
                models.append(
                    ImageRewardWrapper(
                        model_name=args.image_reward_name,
                        device=device,
                        cache_dir=args.image_reward_cache,
                    )
                )
            except RuntimeError as exc:
                logging.warning("Skipping ImageReward model: %s", exc)
        else:
            logging.warning("Unknown model '%s' requested; skipping", model_name)

    if not models:
        raise ValueError("No reward models were instantiated. Please check --models.")

    return models


def evaluate_models(
    prompts: Sequence[PromptRecord],
    models: Sequence[RewardModel],
    limit: Optional[int] = None,
) -> Dict[str, object]:
    per_prompt_results: List[Dict[str, object]] = []
    summary_stats: Dict[str, Dict[str, List[float]]] = {
        model.name: {metric: [] for metric in SUMMARY_METRICS}
        for model in models
    }

    for idx, prompt_record in enumerate(tqdm(prompts, desc="Evaluating prompts", unit="prompt")):
        try:
            profile = build_profile(prompt_record)
        except ValueError as exc:
            logging.warning(
                "Skipping prompt %s: %s",
                prompt_record.prompt_index if prompt_record.prompt_index is not None else idx,
                exc,
            )
            continue

        human_order = consensus_order(profile)
        condorcet = _coerce_single_winner(profile.condorcet_winner())
        borda_winners = sorted(set(borda(profile)))
        plurality_winners = sorted(set(plurality(profile)))
        copeland_winners = sorted(set(copeland(profile)))

        prompt_identifier = (
            prompt_record.prompt_index if prompt_record.prompt_index is not None else idx
        )

        prompt_result = {
            "prompt_index": prompt_identifier,
            "prompt": prompt_record.prompt,
            "image_filenames": prompt_record.image_filenames,
            "human_consensus": {
                "order_indices": human_order,
                "order_filenames": [
                    prompt_record.image_filenames[idx] for idx in human_order
                ],
                "condorcet_winner": condorcet,
                "winner_indices": borda_winners,
                "winner_filenames": [
                    prompt_record.image_filenames[idx] for idx in borda_winners
                ],
            },
            "voting_rules": {
                "plurality": {
                    "winner_indices": plurality_winners,
                    "winner_filenames": [
                        prompt_record.image_filenames[idx] for idx in plurality_winners
                    ],
                },
                "copeland": {
                    "winner_indices": copeland_winners,
                    "winner_filenames": [
                        prompt_record.image_filenames[idx] for idx in copeland_winners
                    ],
                },
            },
            "models": {},
        }

        for model in models:
            try:
                ranking_indices, scores = model.rank(
                    prompt_record.prompt,
                    prompt_record.image_paths,
                )
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"{model.name} failed to open an image for prompt {prompt_identifier}: {exc}.\n"
                    "Verify that image paths are correct or supply --image-root."
                ) from exc

            metrics = compute_metrics(
                human_order,
                ranking_indices,
                condorcet,
                borda_winners,
                plurality_winners,
                copeland_winners,
            )
            prompt_result["models"][model.name] = {
                "scores": scores,
                "order_indices": ranking_indices,
                "order_filenames": [
                    prompt_record.image_filenames[idx] for idx in ranking_indices
                ],
                "metrics": metrics,
            }

            for metric_name, metric_value in metrics.items():
                if metric_value is not None and not math.isnan(metric_value):
                    summary_stats[model.name][metric_name].append(metric_value)

        per_prompt_results.append(prompt_result)

        if limit is not None and len(per_prompt_results) >= limit:
            break

    aggregated_summary: Dict[str, Dict[str, Optional[float]]] = {}
    for model_name, metrics in summary_stats.items():
        aggregated_summary[model_name] = {
            metric: float(mean(values)) if values else None
            for metric, values in metrics.items()
        }
        aggregated_summary[model_name]["prompts_evaluated"] = len(per_prompt_results)

    return {
        "per_prompt": per_prompt_results,
        "summary": aggregated_summary,
    }


def save_summary_csv(summary: Dict[str, Dict[str, object]], base_output: Path) -> None:
    """Persist aggregate metrics as a CSV table alongside the JSON output."""

    csv_path = base_output.with_suffix(".csv")
    fieldnames = ["model", *SUMMARY_METRICS, "prompts_evaluated"]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model_name, metrics in summary.items():
            row = {"model": model_name}
            for metric in SUMMARY_METRICS:
                value = metrics.get(metric)
                if isinstance(value, (int, float)):
                    row[metric] = f"{value:.6f}"
                elif value is None:
                    row[metric] = ""
                else:
                    row[metric] = str(value)
            row["prompts_evaluated"] = metrics.get("prompts_evaluated", "")
            writer.writerow(row)

    logging.info("Saved summary CSV to %s", csv_path)


def save_per_prompt_csv(per_prompt: List[Dict[str, object]], base_output: Path) -> None:
    """Flatten the per-prompt results into a CSV for quick inspection."""

    csv_path = base_output.with_name(f"{base_output.stem}_per_prompt.csv")
    fieldnames = [
        "prompt_index",
        "prompt",
        "model",
        "order_indices",
        "order_filenames",
        *SUMMARY_METRICS,
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in per_prompt:
            base_info = {
                "prompt_index": entry.get("prompt_index"),
                "prompt": entry.get("prompt"),
            }
            for model_name, model_payload in entry.get("models", {}).items():
                row = dict(base_info)
                row["model"] = model_name
                row["order_indices"] = " ".join(str(x) for x in model_payload.get("order_indices", []))
                row["order_filenames"] = " ".join(model_payload.get("order_filenames", []))
                metrics = model_payload.get("metrics", {})
                for metric in SUMMARY_METRICS:
                    value = metrics.get(metric)
                    if isinstance(value, (int, float)):
                        row[metric] = f"{value:.6f}"
                    elif value is None:
                        row[metric] = ""
                    else:
                        row[metric] = str(value)
                writer.writerow(row)

    logging.info("Saved per-prompt CSV to %s", csv_path)


def render_metric_plots(summary: Dict[str, Dict[str, object]], base_output: Path) -> None:
    """Create simple bar charts for each metric if matplotlib is available."""

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        logging.warning("matplotlib not installed; skipping metric plots")
        return

    for metric in SUMMARY_METRICS:
        values: List[float] = []
        labels: List[str] = []
        for model_name, metrics in summary.items():
            value = metrics.get(metric)
            if isinstance(value, (int, float)):
                labels.append(model_name)
                values.append(value)

        if not values:
            continue

        width = max(6.0, len(labels) * 1.6)
        fig, ax = plt.subplots(figsize=(width, 4.5))
        positions = range(len(labels))
        ax.bar(positions, values, color="#4C72B0")
        ax.set_xticks(list(positions))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{metric.replace('_', ' ').title()} by Reward Model")
        for pos, val in zip(positions, values):
            ax.text(pos, val, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()

        plot_path = base_output.with_name(f"{base_output.stem}_{metric}.png")
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)
        logging.info("Saved %s plot to %s", metric, plot_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark image reward models against HPDv2 human rankings.")
    parser.add_argument(
        "--dataset",
        default=Path("ranking_models/test.json"),
        type=Path,
        help="Path to HPDv2 rankings JSON/JSONL file",
    )
    parser.add_argument(
        "--image-root",
        default=None,
        type=Path,
        help="Optional directory used to resolve relative image paths",
    )
    parser.add_argument(
        "--output",
        default=Path("ranking_results/reward_model_benchmark.json"),
        type=Path,
        help="Where to store the JSON results",
    )
    parser.add_argument("--models", nargs="+", default=["open_clip"], help="List of reward models to run (open_clip, hpsv2, image_reward)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device")
    parser.add_argument("--precision", choices=["fp32", "amp"], default="amp", help="Computation precision for CLIP-based models")
    parser.add_argument("--max-prompts", type=int, default=None, help="Optional cap on number of prompts to evaluate (for quick tests)")

    # OpenCLIP settings
    parser.add_argument("--openclip-model", default="ViT-H-14", help="OpenCLIP model backbone")
    parser.add_argument("--openclip-pretrained", default="laion2b_s32b_b79k", help="OpenCLIP pretrained weights tag")

    # HPSv2 settings
    parser.add_argument("--hpsv2-version", default="v2.1", help="HPSv2 model version to use (e.g., v2 or v2.1)")

    # ImageReward settings
    parser.add_argument("--image-reward-name", default="ImageReward-v1.0", help="ImageReward model identifier")
    parser.add_argument("--image-reward-cache", default=None, help="Optional cache directory for ImageReward downloads")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    device = torch.device(args.device)
    data = load_dataset(args.dataset, max_prompts=args.max_prompts, image_root=args.image_root)
    logging.info("Loaded %d prompts from %s", len(data), args.dataset)
    if args.image_root is not None:
        logging.info("Resolving images relative to %s", args.image_root)
    models = prepare_models(args, device)

    logging.info("Running benchmark with %d prompts on models: %s", len(data), ", ".join(m.name for m in models))
    results = evaluate_models(data, models, limit=args.max_prompts)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    save_summary_csv(results["summary"], args.output)
    save_per_prompt_csv(results["per_prompt"], args.output)
    render_metric_plots(results["summary"], args.output)

    logging.info("Saved benchmark results to %s", args.output)
    logging.info("Summary statistics:")
    for model_name, metrics in results["summary"].items():
        metric_str_parts = []
        for metric_name, metric_value in metrics.items():
            if metric_name == "prompts_evaluated":
                continue
            if isinstance(metric_value, (int, float)):
                metric_str_parts.append(f"{metric_name}={metric_value:.3f}")
            elif metric_value is None:
                metric_str_parts.append(f"{metric_name}=n/a")
        metric_str = ", ".join(metric_str_parts)
        logging.info("  %s -> %s", model_name, metric_str)


if __name__ == "__main__":
    main()
