import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

_LOG = logging.getLogger("harness.skill")

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    yaml = None
    _YAML_AVAILABLE = False

# 技能激活时注入的引导文本最大长度（字符数）
MAX_ACTIVATION_GUIDE_LENGTH = 2000

@dataclass
class SkillStage:
    name: str
    description: str          # 某一步的描述
    allowed_tools: Set[str]   # 该阶段允许的工具名集合（"*" = 全部）
    stage_guide: str = ""     # 该阶段的引导文本，例如使用示例

@dataclass
class SkillDefinition:
    name: str
    description: str
    global_guide: str
    is_sop: bool
    initial_stage: str        # 初始步骤
    stages: Dict[str, SkillStage] = field(default_factory=dict)
    allowed_transitions: Dict[str, List[str]] = field(default_factory=dict)
    terminal_stages: Set[str] = field(default_factory=set)

    def validate(self) -> None:
        if self.initial_stage not in self.stages:
            raise ValueError(f"Initial stage '{self.initial_stage}' not in stages.")

        # transitions 目标必须是已声明的 stage，拼错尽早暴露
        for src, targets in self.allowed_transitions.items():
            if src not in self.stages:
                raise ValueError(f"Transition source '{src}' not in stages.")
            for tgt in targets:
                if tgt not in self.stages:
                    raise ValueError(
                        f"Transition target '{tgt}' (from '{src}') not in stages.")

        all_sources = set(self.allowed_transitions.keys())
        # 终止步骤 = 所有步骤 - 有出边的步骤
        self.terminal_stages = set(self.stages.keys()) - all_sources


class ProgressiveSkillManager:
    def __init__(self, skills_dir: str = "./skills"):
        self.skills_dir = skills_dir
        self.skills_pool: Dict[str, SkillDefinition] = {}

    # ------------------------------------------------------------
    # 加载所有技能
    # ------------------------------------------------------------
    def load_all_skills(self):
        if not os.path.exists(self.skills_dir):
            _LOG.warning("skills 目录不存在: %s", self.skills_dir)
            return

        # 先收集所有解析结果，再统一注册，以便检测重名
        parsed_skills: List[SkillDefinition] = []
        failed = 0
        for folder in os.listdir(self.skills_dir):
            # 兼容 SKILL.md (大写) 和 skill.md (小写)
            for md_file in ("SKILL.md", "skill.md"):
                md_path = os.path.join(self.skills_dir, folder, md_file)
                if os.path.isfile(md_path):
                    try:
                        with open(md_path, "r", encoding="utf-8") as f:
                            skill_def = self._parse_skill_file(f.read(), folder)
                        parsed_skills.append(skill_def)
                    except Exception:
                        failed += 1
                        _LOG.error("技能加载失败: %s（已跳过）", md_path, exc_info=True)
                    break  # 每个文件夹只加载一个文件

        # 统一注册（检测重名）
        for skill in parsed_skills:
            if skill.name in self.skills_pool:
                raise ValueError(
                    f"技能重名: '{skill.name}' 已存在。"
                    f"请检查技能目录并确保名称唯一。"
                )
            self.skills_pool[skill.name] = skill
            _LOG.info("[SkillManager] 注册技能: %s (SOP=%s)", skill.name, skill.is_sop)

        _LOG.info("技能加载完成：成功 %d，失败 %d", len(self.skills_pool), failed)

    # ------------------------------------------------------------
    # 解析单个技能文件
    # ------------------------------------------------------------
    def _parse_skill_file(self, content: str, fallback_name: str) -> SkillDefinition:
        meta: Dict = {}
        global_guide = content
        is_sop = False
        stages_data: Dict = {}
        transitions_data: Dict = {}

        # ---------- 尝试解析 YAML Frontmatter ----------
        yaml_match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, re.DOTALL)
        yaml_parsed = False
        if yaml_match and _YAML_AVAILABLE:
            try:
                parsed = yaml.safe_load(yaml_match.group(1)) or {}
                if isinstance(parsed, dict):
                    meta = parsed
                    yaml_parsed = True
                    global_guide = yaml_match.group(2).strip()
                    is_sop = "stages" in meta
                    if is_sop:
                        stages_data = meta.get("stages", {}) or {}
                        transitions_data = meta.get("transitions", {}) or {}
            except yaml.YAMLError:
                _LOG.warning("YAML frontmatter 解析失败，将尝试其他方式")

        # ---------- 如果 YAML 未提供 name/description，尝试 Markdown 表格 ----------
        if not meta.get("name"):
            name_match = re.search(r"\|\s*name\s*\|\s*(.*?)\s*\|", content, re.IGNORECASE)
            desc_match = re.search(r"\|\s*description\s*\|\s*(.*?)\s*\|", content, re.IGNORECASE)
            if name_match:
                meta["name"] = name_match.group(1).strip()
            if desc_match:
                meta["description"] = desc_match.group(1).strip()
            # 如果没有 YAML 解析，且走了表格路径，需要清洗 global_guide
            if not yaml_parsed:
                # 去掉表格行和可能的 YAML 块
                global_guide = self._clean_guide_text(content)

        # ---------- 兜底 name / description ----------
        if not meta.get("name"):
            meta["name"] = fallback_name
        if not meta.get("description"):
            lines = [l.strip() for l in content.split("\n")
                     if l.strip() and not l.startswith("#")
                     and not l.startswith("|") and not l.startswith("---")]
            meta["description"] = lines[0][:100] if lines else "No description provided."

        # ---------- 构建 SkillDefinition ----------
        skill_def = SkillDefinition(
            name=meta["name"],
            description=meta.get("description", ""),
            global_guide=global_guide,
            is_sop=is_sop,
            initial_stage="",
        )

        if is_sop:
            for s_name, s_data in stages_data.items():
                s_data = s_data or {}
                skill_def.stages[s_name] = SkillStage(
                    name=s_name,
                    description=s_data.get("description", ""),
                    allowed_tools=set(s_data.get("allowed_tools", []) or []),
                    stage_guide=s_data.get("guide", ""),
                )
            # transitions 取值规整为 list
            for src, targets in (transitions_data or {}).items():
                if isinstance(targets, str):
                    targets = [targets]
                skill_def.allowed_transitions[src] = list(targets) if targets else []

            # 优先采用 YAML 显式声明的 initial_stage
            declared_initial = meta.get("initial_stage")
            if declared_initial:
                skill_def.initial_stage = str(declared_initial)
            elif skill_def.stages:
                skill_def.initial_stage = next(iter(skill_def.stages))
        else:
            # 软约束技能统一包装为单阶段状态机
            skill_def.initial_stage = "active_mode"
            skill_def.stages["active_mode"] = SkillStage(
                name="active_mode",
                description="General execution mode relies on Prompt with soft constraints",
                allowed_tools={"*"},
            )

        skill_def.validate()
        return skill_def

    # ------------------------------------------------------------
    # 清洗引导文本：移除表格行、YAML 块等
    # ------------------------------------------------------------
    @staticmethod
    def _clean_guide_text(content: str) -> str:
        # 移除 YAML frontmatter 块
        clean = re.sub(r"^---\s*\n.*?\n---\s*\n", "", content, flags=re.DOTALL)
        # 移除以 | 开头的表格行
        clean = "\n".join(line for line in clean.split("\n")
                          if not line.strip().startswith("|"))
        # 移除多余空行
        clean = re.sub(r"\n{3,}", "\n\n", clean)
        return clean.strip()

    # ------------------------------------------------------------
    # 查询接口
    # ------------------------------------------------------------
    def get_allowed_tools_for_stage(self, skill_name: str, stage_name: str) -> Set[str]:
        skill = self.skills_pool.get(skill_name)
        if not skill:
            return set()
        stage = skill.stages.get(stage_name)
        return stage.allowed_tools if stage else set()

    def get_discovery_system_prompt_patch(self) -> str:
        """技能目录（仅名称+描述），提示 LLM 先 activate_skill 再使用。"""
        if not self.skills_pool:
            return ""
        prompt = ("\n### Available skill library "
                  "(must be activated by calling `activate_skill` before use):\n")
        for name, skill in self.skills_pool.items():
            prompt += f"- **{name}**: {skill.description}\n"
        return prompt

    def activate_skill(self, skill_name: str) -> str:
        skill = self.skills_pool.get(skill_name)
        if not skill:
            return f"[Error] skill not found: {skill_name}"
        guide = skill.global_guide
        if len(guide) > MAX_ACTIVATION_GUIDE_LENGTH:
            guide = guide[:MAX_ACTIVATION_GUIDE_LENGTH] + "\n...(truncated)"
        return (f"[system info] skill {skill_name} activated!\n"
                f"Please strictly follow the guide below:\n\n{guide}")

    def get_initial_stage(self, skill_name: str) -> Optional[str]:
        skill = self.skills_pool.get(skill_name)
        return skill.initial_stage if skill else None

    def get_valid_transitions(self, skill_name: str, stage_name: str) -> List[str]:
        skill = self.skills_pool.get(skill_name)
        if not skill:
            return []
        return skill.allowed_transitions.get(stage_name, [])

