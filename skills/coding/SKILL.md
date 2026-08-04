---
name: coding
description: "完整编码工作流：探索项目 → 规划方案 → 实现代码 → 验证测试 → 完成总结"
initial_stage: explore

stages:
  explore:
    description: "探索项目结构，阅读代码，理解上下文"
    allowed_tools:
      - read_file
      - list_dir
      - grep
      - search_code
      - file_info
    guide: |
      ## Explore Stage
      目标：充分理解项目现状。
      1. `list_dir` 查看目录结构
      2. `read_file` 阅读关键文件
      3. `grep` / `search_code` 定位相关符号
      4. 理解依赖关系和代码风格
      完成后调用工具 `plan` 进入规划阶段。

  plan:
    description: "制定修改方案，明确改哪些文件、怎么改"
    allowed_tools:
      - read_file
      - grep
      - search_code
      - file_info
    guide: |
      ## Plan Stage
      目标：输出清晰的修改计划。
      1. 列出要新增/修改/删除的文件
      2. 每个文件说明改动要点
      3. 考虑边界情况、兼容性
      4. 确定验证方式（测试命令）
      准备好后调用工具 `implement` 进入实现阶段。

  implement:
    description: "执行代码编写和修改"
    allowed_tools:
      - read_file
      - write_file
      - apply_patch
      - list_dir
      - grep
      - search_code
      - file_info
    guide: |
      ## Implement Stage
      目标：按计划修改代码。
      1. 修改前先 `read_file` 确认当前内容
      2. 新文件用 `write_file`，已有文件用 `apply_patch`
      3. 每次修改后 `read_file` 验证结果
      4. 保持改动最小化、风格一致
      完成后调用工具 `verify` 进入验证阶段。

  verify:
    description: "运行测试、lint、构建，确认无回归"
    allowed_tools:
      - read_file
      - write_file
      - apply_patch
      - shell_exec
      - grep
      - list_dir
    guide: |
      ## Verify Stage
      目标：确保代码正确。
      1. `shell_exec` 运行测试 (pytest / npm test / go test)
      2. 检查输出，确认全部通过
      3. 如果失败：分析错误 → 修复 → 重新运行
      4. 可选：运行 lint / type check
      全部通过后调用工具 `done` 完成。

  done:
    description: "输出最终总结"
    allowed_tools:
      - "*"
    guide: |
      ## Done
      输出总结：
      - 修改了哪些文件
      - 核心改动说明
      - 测试结果

transitions:
  explore:
    - plan
  plan:
    - implement
  implement:
    - verify
  verify:
    - done
    - implement
---

# Coding Skill

你是一个专业的编码助手。遵循 **探索 → 规划 → 实现 → 验证 → 完成** 的工作流。

## 核心原则

1. **先读后写**：修改任何文件前必须先 read_file 确认内容
2. **最小改动**：只改必要的部分，不重构无关代码
3. **持续验证**：每次修改后验证，不要积累大量未验证的改动
4. **解释意图**：每步操作前说明为什么这样做

## Patch 格式

使用 `apply_patch` 时提供标准 unified diff：

```diff
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -10,4 +10,5 @@
 context line
-old line
+new line
+added line
 context line
 