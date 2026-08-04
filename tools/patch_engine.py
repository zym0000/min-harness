"""
Patch Engine: 提供更高层的编辑抽象
支持: 行替换、插入、删除、搜索替换
"""
import re
import difflib
from typing import List, Tuple, Optional

class PatchEngine:
    """高级 Patch 引擎，将简单编辑操作转换为 unified diff"""

    @staticmethod
    def generate_replace_patch(file_path: str, original: str,
                                old_text: str, new_text: str) -> str:
        """生成搜索替换的 patch"""
        if old_text not in original:
            # 尝试模糊匹配
            fuzzy_result = PatchEngine._fuzzy_find(original, old_text)
            if fuzzy_result is None:
                raise ValueError(
                    f"Text not found in file.\n"
                    f"Searched:\n---\n{old_text[:200]}\n---"
                )
            matched_text, ratio = fuzzy_result
            # 可以记录日志或返回警告，这里仅采用模糊匹配的结果
            old_text = matched_text

        new_content = original.replace(old_text, new_text, 1)
        return PatchEngine._make_diff(file_path, original, new_content)

    @staticmethod
    def generate_line_replace_patch(file_path: str, original: str,
                                     start_line: int, end_line: int,
                                     new_lines: str) -> str:
        """生成行范围替换的 patch"""
        lines = original.splitlines(keepends=True)
        new_line_list = new_lines.splitlines(keepends=True)

        result = lines[:start_line - 1] + new_line_list + lines[end_line:]
        new_content = "".join(result)
        return PatchEngine._make_diff(file_path, original, new_content)

    @staticmethod
    def generate_insert_patch(file_path: str, original: str,
                               line_number: int, insert_text: str) -> str:
        """在指定行后插入内容"""
        lines = original.splitlines(keepends=True)
        insert_lines = insert_text.splitlines(keepends=True)
        result = lines[:line_number] + insert_lines + lines[line_number:]
        new_content = "".join(result)
        return PatchEngine._make_diff(file_path, original, new_content)

    @staticmethod
    def generate_delete_patch(file_path: str, original: str,
                               start_line: int, end_line: int) -> str:
        """删除指定行范围"""
        lines = original.splitlines(keepends=True)
        result = lines[:start_line - 1] + lines[end_line:]
        new_content = "".join(result)
        return PatchEngine._make_diff(file_path, original, new_content)

    @staticmethod
    def _make_diff(file_path: str, original: str, new_content: str) -> str:
        orig_lines = original.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        diff = difflib.unified_diff(
            orig_lines, new_lines,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            lineterm="",          # 修复：避免重复换行
        )
        return "".join(diff)

    @staticmethod
    def _fuzzy_find(content: str, target: str, threshold: float = 0.8) -> Optional[Tuple[str, float]]:
        """
        模糊查找：当精确匹配失败时尝试。
        返回 (匹配文本, 相似度) 或 None。
        """
        target_lines = target.strip().splitlines()
        content_lines = content.splitlines()

        best_match = None
        best_ratio = 0.0

        for i in range(len(content_lines) - len(target_lines) + 1):
            window = "\n".join(content_lines[i:i + len(target_lines)])
            ratio = difflib.SequenceMatcher(None, target.strip(), window.strip()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = window

        if best_ratio >= threshold:
            return best_match, best_ratio
        return None

    @staticmethod
    def validate_patch(original: str, patch_text: str) -> Tuple[bool, str]:
        """验证 patch 是否可以干净地应用"""
        lines = original.splitlines()
        total_lines = len(lines)

        hunk_pattern = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
        for match in hunk_pattern.finditer(patch_text):
            start = int(match.group(1))
            count = int(match.group(2)) if match.group(2) else 1
            if start + count - 1 > total_lines:
                return False, f"Hunk at line {start} exceeds file length ({total_lines})"

        return True, "Patch looks valid"