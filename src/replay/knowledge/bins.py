"""按论文提示词离线提取并缓存语义类别。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch

from replay.core.config import ReplayConfig, write_json

if TYPE_CHECKING:
    from replay.knowledge.encoders import SharedKB
    from replay.knowledge.store import PosteriorKB


SYSTEM_PROMPT = (
    "You are an expert in visual scene analysis. You have access to external knowledge retrieved "
    "from a database using retrieve(·). Use the retrieved context below as your primary source "
    "of truth. Your task is to identify concrete objects that cameras, LiDAR, or radar sensors "
    "would detect in real-world scenarios. List 4–8 specific objects, people, or physical "
    "elements that sensors would commonly observe in user's task scenario. Focus on visible, "
    "tangible items that appear in sensor data. Return only the object names, separated by commas."
)


def _normalize_name(value: str) -> str:
    """统一大小写、连续空白、连字符和下划线，避免同名候选重复计数。"""
    return re.sub(r"[\s_-]+", " ", value.strip().casefold()).strip()


def _parse_candidates(response: str) -> list[str]:
    """只清理模型原文的列表格式；保留所有实际出现的候选。"""
    parts = re.split(r"[,，;；\n]+", response)
    bins: list[str] = []
    for part in parts:
        cleaned = re.sub(r"^\s*[-*\d.)]+\s*", "", part).strip().strip('"\'.。!！?？:： ')
        name = _normalize_name(cleaned)
        if name and name not in bins:
            bins.append(name)
    return bins


def parse_bins(response: str, minimum: int, maximum: int) -> list[str]:
    # 未配置必需类别的其他任务也遵守论文的 4–8 个类别范围。
    bins = _parse_candidates(response)
    if not minimum <= len(bins) <= maximum:
        raise ValueError(f"模型输出 {len(bins)} 个类别，不在论文要求的 {minimum}–{maximum} 范围：{response!r}")
    return bins


def canonicalize_bins(candidates: list[str], aliases: dict[str, str]) -> list[str]:
    """把模型确实提到的同义词归一；绝不生成原候选中没有的类别。"""
    normalized_aliases = {
        _normalize_name(key): _normalize_name(value)
        for key, value in aliases.items()
    }
    canonical: list[str] = []
    for candidate in candidates:
        name = _normalize_name(candidate)
        name = normalized_aliases.get(name, name)
        if name and name not in canonical:
            canonical.append(name)
    return canonical


def select_required_bins(response: str, required_bins: list[str], allowed_bins: list[str],
                         aliases: dict[str, str]) -> dict:
    """生成可审计的模型候选与最终类别；缺失项由调用方报错，不补齐。"""
    raw_candidates = _parse_candidates(response)
    canonical_candidates = canonicalize_bins(raw_candidates, aliases)
    missing = [name for name in required_bins if name not in canonical_candidates]
    # 按配置顺序输出，但每一项必须能追溯到模型原始候选。
    selected = [name for name in allowed_bins if name in canonical_candidates]
    return {
        "raw_candidates": raw_candidates,
        "canonical_candidates": canonical_candidates,
        "excluded_candidates": [name for name in canonical_candidates if name not in allowed_bins],
        "required_bins": list(required_bins),
        "allowed_bins": list(allowed_bins),
        "missing_required_bins": missing,
        "selected_bins": selected,
    }


def select_required_bins_with_retry(response: str, minimum: int, maximum: int,
                                    required_bins: list[str], allowed_bins: list[str],
                                    aliases: dict[str, str], retry: Callable[[str], str]) -> dict:
    """原始与最终类别均须符合 4–8 范围；失败时由同一模型重试一次。"""
    outputs = [response]
    attempts = []
    # 模型最多重试一次格式/类别约束；仍缺类别时保留失败记录，不补造类别。
    for turn in range(2):
        selection = select_required_bins(outputs[-1], required_bins, allowed_bins, aliases)
        count = len(selection["raw_candidates"])
        selected_count = len(selection["selected_bins"])
        in_range = minimum <= count <= maximum
        selected_in_range = minimum <= selected_count <= maximum
        accepted = (in_range and selected_in_range and
                    not selection["missing_required_bins"])
        attempts.append({"raw_model_output": outputs[-1], "raw_count": count,
                         "raw_count_in_paper_range": in_range,
                         "selected_count": selected_count,
                         "selected_count_in_paper_range": selected_in_range,
                         "accepted": accepted, **selection})
        if accepted or turn == 1:
            break
        outputs.append(retry(outputs[-1]))
    return {"raw_model_output": outputs[-1], "raw_model_outputs": outputs,
            "format_retry_project_choice": len(outputs) > 1,
            "attempts": attempts, "selection": selection,
            "raw_count_in_paper_range": in_range,
            "selected_count_in_paper_range": selected_in_range,
            "accepted": accepted}


def _configured_constraints(task: dict) -> tuple[list[str], list[str], dict[str, str]] | None:
    """只对显式配置的任务启用必需类别策略，不改变其他任务。"""
    required, allowed = task.get("required_bins"), task.get("allowed_bins")
    if required is None and allowed is None:
        return None
    if not isinstance(required, list) or not isinstance(allowed, list) or not required or not allowed:
        raise ValueError("required_bins 和 allowed_bins 必须同时配置为非空列表")
    if (not all(isinstance(name, str) and name.strip() == name and name == name.lower()
                for name in required + allowed)
            or len(set(required)) != len(required) or len(set(allowed)) != len(allowed)
            or not set(required).issubset(allowed)):
        raise ValueError("必需/允许类别须为去重的小写名称，且必需类别属于允许集合")
    aliases = task.get("bin_aliases", {})
    if not isinstance(aliases, dict) or not all(
            isinstance(key, str) and key.strip() and isinstance(value, str) and value in allowed
            for key, value in aliases.items()):
        raise ValueError("bin_aliases 必须把非空同义词映射到 allowed_bins 中的标准类别")
    return required, allowed, aliases


def _load_qwen(config: ReplayConfig, device: torch.device):
    """两个离线命令共用已锁定的 Qwen 权重；不会改动接收端模型。"""
    from replay.core.assets import verify_model_lock
    from transformers import AutoModelForCausalLM, AutoTokenizer

    verify_model_lock(config, "qwen")
    model_path = config.model_path("qwen")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    inference_dtype = (torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported()
                       else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=inference_dtype).to(device).eval()
    return tokenizer, model


def _qwen_answer(tokenizer, model, device: torch.device, messages: list[dict],
                 max_new_tokens: int = 128) -> str:
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    encoded = tokenizer(rendered, return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(output[0, encoded.input_ids.shape[1]:], skip_special_tokens=True)


_VISUAL_QUERY_SYSTEM = (
    "You write short English object-name queries for a visual region proposal model. "
    "For the target category, return 0 to 3 alternative names for the same visible kind of object. "
    "A common visual name is useful when the target name is specialized. "
    "Do not return other task categories, subtypes, scene descriptions, properties, actions, "
    "or the target name itself. If there is no safe equivalent, return []. "
    "Return only a JSON array of strings."
)


def _parse_visual_aliases(response: str) -> list[str]:
    """容忍 Qwen 的代码围栏/列表格式，拒绝描述句与非英文查询。"""
    values: list[str] | None = None
    bracket = response.find("[")
    if bracket >= 0:
        try:
            parsed, _ = json.JSONDecoder().raw_decode(response[bracket:])
            if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
                values = parsed
        except json.JSONDecodeError:
            pass
    if values is None:
        values = re.split(r"[,，;；\n]+", response)
    aliases: list[str] = []
    for value in values:
        cleaned = re.sub(r"^\s*[-*\d.)]+\s*", "", value).strip().strip('"\'.。!！?？:： `')
        name = _normalize_name(cleaned)
        # 只允许最多三个英文词的短名词短语，避免整句误作 CLIPSeg 提示词。
        if (re.fullmatch(r"[a-z]+(?: [a-z]+){0,2}", name)
                and not any(word in {"in", "on", "with", "near", "under", "behind", "beside"}
                            for word in name.split())
                and name not in aliases):
            aliases.append(name)
    return aliases[:3]


def build_visual_queries(bins: list[str], task_description: str, shared: SharedKB,
                         generate: Callable[[str, list[str], str], str],
                         similarity_floor: float = 0.45,
                         ambiguity_margin: float = 0.02) -> tuple[dict[str, list[str]], dict]:
    """离线生成桶的视觉查询；CLIP 校验只影响区域候选，不修改语义桶。"""
    if not bins or len({_normalize_name(name) for name in bins}) != len(bins):
        raise ValueError("语义桶须为非空且不重复")
    if not -1 <= similarity_floor <= 1 or not 0 <= ambiguity_margin <= 2:
        raise ValueError("视觉查询相似度校验参数无效")

    canonical = [_normalize_name(name) for name in bins]
    raw_outputs: dict[str, str] = {}
    proposed: dict[str, list[str]] = {}
    for name in canonical:
        other = [item for item in canonical if item != name]
        raw_outputs[name] = generate(name, other, task_description)
        proposed[name] = [alias for alias in _parse_visual_aliases(raw_outputs[name]) if alias != name]

    all_texts = list(dict.fromkeys(canonical + [alias for names in proposed.values() for alias in names]))
    vectors = np.asarray(shared.text(all_texts), dtype=np.float32)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    lookup = dict(zip(all_texts, vectors))
    queries = {name: [name] for name in canonical}
    audit: dict = {
        "method": "qwen_visual_aliases_clip_text_validation_project_choice",
        "similarity_floor": similarity_floor,
        "ambiguity_margin": ambiguity_margin,
        "raw_outputs": raw_outputs,
        "candidates": {},
    }
    for name in canonical:
        decisions = []
        for alias in proposed[name]:
            own = float(lookup[alias] @ lookup[name])
            rivals = {other: float(lookup[alias] @ lookup[other])
                      for other in canonical if other != name}
            nearest_rival = max(rivals.values(), default=-1.0)
            if alias in canonical:
                reason = "another_canonical_bin"
            elif name in alias.split():
                # 例如 red car 是 car 的子类/属性组合，不是类别等价名。
                reason = "target_with_modifier"
            elif own < similarity_floor:
                reason = "low_clip_similarity"
            elif own < nearest_rival + ambiguity_margin:
                reason = "ambiguous_across_bins"
            else:
                reason = "accepted"
                queries[name].append(alias)
            decisions.append({"query": alias, "own_cosine": own,
                              "max_other_cosine": nearest_rival, "reason": reason})
        audit["candidates"][name] = decisions

    # 同一个别名不能同时为两个桶提区域；原名永远保留。
    counts = {alias: sum(alias in names[1:] for names in queries.values())
              for names in queries.values() for alias in names[1:]}
    for name in canonical:
        duplicates = {alias for alias in queries[name][1:] if counts[alias] > 1}
        if duplicates:
            queries[name] = [name] + [alias for alias in queries[name][1:] if alias not in duplicates]
            for decision in audit["candidates"][name]:
                if decision["query"] in duplicates:
                    decision["reason"] = "duplicate_across_bins"
    audit["visual_queries"] = queries
    return queries, audit


def _generate_visual_queries(config: ReplayConfig, shared: SharedKB, device: torch.device,
                             bins: list[str], tokenizer, model) -> tuple[dict[str, list[str]], dict]:
    def generate(name: str, others: list[str], description: str) -> str:
        prompt = (f"Task: {description}\nTarget category: {name}\n"
                  f"Other categories (do not return these): {', '.join(others) or 'none'}\n"
                  "Alternative visual object names (JSON array):")
        return _qwen_answer(tokenizer, model, device, [
            {"role": "system", "content": _VISUAL_QUERY_SYSTEM},
            {"role": "user", "content": prompt},
        ], max_new_tokens=96)

    return build_visual_queries(bins, str(config.section("task")["description"]), shared, generate)


def enrich_visual_queries(config: ReplayConfig, shared: SharedKB, device: torch.device,
                          *, cache_path: Path | None = None) -> dict:
    """升级旧语义桶缓存，只补视觉查询，不重新生成或变更 bins。"""
    path = cache_path or config.path(config.section("task")["bin_cache"])
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("task") != config.section("task")["description"]:
        raise ValueError("缓存的任务描述与当前配置不一致")
    bins = data.get("bins")
    if not isinstance(bins, list) or not bins or not all(isinstance(name, str) for name in bins):
        raise ValueError("旧缓存缺少有效语义桶")
    if "visual_queries" in data:
        return data
    tokenizer, model = _load_qwen(config, device)
    queries, audit = _generate_visual_queries(config, shared, device, bins, tokenizer, model)
    data["visual_queries"] = queries
    data["visual_query_audit"] = audit
    data["visual_query_qwen_revision"] = config.section("models")["qwen"]["revision"]
    data["visual_query_clip_revision"] = config.section("models")["clip"]["revision"]
    write_json(path, data)
    return data


def extract_bins(config: ReplayConfig, shared: SharedKB, kb: PosteriorKB | None, device: torch.device) -> dict:
    from replay.core.assets import verify_model_lock
    from transformers import AutoModelForCausalLM, AutoTokenizer
    verify_model_lock(config, "qwen")
    task = config.section("task")
    constraints = _configured_constraints(task)
    text = str(task["description"])
    rag_top_k = int(config.section("knowledge")["rag_top_k"])
    query: np.ndarray = shared.text([text])[0]
    # 场景词表定义和参考图像共存于 KB；先按来源筛选可避免同类图像记录
    # 在离线类别生成时占满 top-k。来源筛选不是论文公开参数，只由场景配置启用。
    rag_sources = config.section("knowledge").get("rag_sources")
    if kb is None:
        references = []
    elif rag_sources is None:
        references = kb.search(query, rag_top_k)
    else:
        if not isinstance(rag_sources, list) or not rag_sources or not all(
                isinstance(source, str) for source in rag_sources):
            raise ValueError("rag_sources 必须是非空的来源名称列表")
        allowed = set(rag_sources)
        candidates = kb.search(query, len(kb.records))
        references = [hit for hit in candidates if hit["record"].get("source") in allowed][:rag_top_k]
    context = "\n".join(str(hit["record"].get("text", "")) for hit in references)
    prompt = f'Task scenario: "{text}"\nRetrieved context:\n{context}\nObjects:'

    model_path: Path = config.model_path("qwen")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    # 论文未规定 Qwen 推理精度；4090 上用 BF16 避免 4B 权重与 CLIP 同卡时显存溢出。
    inference_dtype = (torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported()
                       else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=inference_dtype).to(device).eval()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    encoded = tokenizer(rendered, return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model.generate(**encoded, max_new_tokens=128, do_sample=False)
    answer = tokenizer.decode(output[0, encoded.input_ids.shape[1]:], skip_special_tokens=True)
    answers = [answer]
    minimum, maximum = int(task["bin_min"]), int(task["bin_max"])
    selection = None
    if constraints is not None:
        required, allowed, aliases = constraints
        def retry_required(previous: str) -> str:
            # 缺少核心类别或同义词合并后不足 4 类，也请模型重选；程序不补造类别。
            previous_selection = select_required_bins(previous, required, allowed, aliases)
            count = len(previous_selection["raw_candidates"])
            selected_count = len(previous_selection["selected_bins"])
            retry_messages = messages + [
                {"role": "assistant", "content": previous},
                {"role": "user", "content": (
                    f"Your previous answer contained {count} distinct names but only "
                    f"{selected_count} unique task categories after synonym normalization. "
                    f"Generate {minimum}–{maximum} distinct visible object categories for {text}. "
                    f"Include these task-critical categories: {', '.join(required)}. "
                    f"Other eligible categories are: {', '.join(name for name in allowed if name not in required)}. "
                    "Avoid synonyms of the same category, road, and buildings. "
                    "Return only category names separated by commas; do not add explanations."
                )},
            ]
            retry_rendered = tokenizer.apply_chat_template(
                retry_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            retry_encoded = tokenizer(retry_rendered, return_tensors="pt").to(device)
            with torch.inference_mode():
                retry_output = model.generate(**retry_encoded, max_new_tokens=128, do_sample=False)
            return tokenizer.decode(retry_output[0, retry_encoded.input_ids.shape[1]:],
                                    skip_special_tokens=True)

        retry_result = select_required_bins_with_retry(
            answer, minimum, maximum, required, allowed, aliases, retry_required)
        answer = retry_result["raw_model_output"]
        answers = retry_result["raw_model_outputs"]
        selection = retry_result["selection"]
        audit = {
            "task": text, "raw_model_output": answer, "raw_model_outputs": answers,
            **selection, "bin_aliases": aliases,
            "raw_count_in_paper_range": retry_result["raw_count_in_paper_range"],
            "selected_count_in_paper_range": retry_result["selected_count_in_paper_range"],
            "format_retry_project_choice": retry_result["format_retry_project_choice"],
            "attempts": retry_result["attempts"],
            "rag_record_ids": [hit["record"].get("id") for hit in references],
        }
        # 导出脚本会重定向 bin_cache；审计跟随该路径，避免多次运行互相覆盖。
        audit_path = config.path(Path(task["bin_cache"]).with_suffix(".audit.json"))
        if not retry_result["raw_count_in_paper_range"]:
            audit["status"] = "rejected_raw_count_out_of_range"
            write_json(audit_path, audit)
            raise ValueError(
                f"模型重试后仍输出 {len(selection['raw_candidates'])} 个类别，"
                f"不在论文要求的 {minimum}–{maximum} 范围")
        if selection["missing_required_bins"]:
            audit["status"] = "rejected_missing_required_bins"
            write_json(audit_path, audit)
            raise ValueError("模型原始候选缺少必需类别：" + ", ".join(selection["missing_required_bins"]))
        if not retry_result["selected_count_in_paper_range"]:
            audit["status"] = "rejected_selected_count_out_of_range"
            write_json(audit_path, audit)
            raise ValueError(
                f"同义词合并和任务类别过滤后只有 {len(selection['selected_bins'])} 个类别，"
                f"不在 {minimum}–{maximum} 范围")
        audit["status"] = "accepted"
        write_json(audit_path, audit)
        bins = selection["selected_bins"]
    else:
        try:
            bins = parse_bins(answer, minimum, maximum)
        except ValueError:
            # 4B 在 19 条类别定义下可能罗列过多；让同一模型自行压缩一次，
            # 不由程序手工挑选或新增类别。此纠错轮次是项目设计，两个原始输出均记录。
            messages += [
                {"role": "assistant", "content": answer},
                {"role": "user", "content": (
                    f"Your previous answer listed too many objects. Choose {minimum}–{maximum} distinct, "
                    f"high-priority categories for {text} from that answer. Cover different "
                    "kinds of road users, traffic controls, and surrounding physical elements; avoid "
                    "listing several similar vehicle subtypes. Return only the names, separated by commas."
                )},
            ]
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                     enable_thinking=False)
            encoded = tokenizer(rendered, return_tensors="pt").to(device)
            with torch.inference_mode():
                output = model.generate(**encoded, max_new_tokens=128, do_sample=False)
            answer = tokenizer.decode(output[0, encoded.input_ids.shape[1]:], skip_special_tokens=True)
            answers.append(answer)
            bins = parse_bins(answer, minimum, maximum)
    result = {
        "task": text,
        "bins": bins,
        "raw_model_output": answer,
        "raw_model_outputs": answers,
        "format_retry_project_choice": len(answers) > 1,
        "selection_audit": selection,
        "rag_record_ids": [hit["record"].get("id") for hit in references],
        "rag_sources_project_choice": sorted(rag_sources) if rag_sources is not None else None,
        "qwen_revision": config.section("models")["qwen"]["revision"],
        "clip_revision": config.section("models")["clip"]["revision"],
    }
    write_json(config.path(task["bin_cache"]), result)
    return result


def read_bins(path: Path, task: str) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["task"] != task:
        raise ValueError("缓存的任务描述与当前配置不一致")
    return list(data["bins"])
