"""push_repo.py — 经 gh api (Git Data API) 推送 oodforge 到 CJX0712/oodforge。

绕开 git 智能协议经代理 502 的坑: 全程走 api.github.com (出口稳定)。
初版为空仓库时用 [parents:[]] 建初始 commit，再 POST git/refs。
含 8 次退避重试, 命中 bad gateway / 502 / 503 / reset 自动重跑。
作者: 晨星
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

REPO = "CJX0712/oodforge"
REPO_DIR = Path(__file__).resolve().parent
DESCRIPTION = "OODForge · 分布外检测与概率校准系统 (CPU/离线可运行, CCOR 自适应路由). 作者 晨星"
AUTHOR = {"name": "晨星", "email": "CJX0712@users.noreply.github.com"}
EXCLUDE_DIRS = {".git", ".venv", "venv", "envs", "__pycache__"}
EXCLUDE_EXT = {".pyc"}

RETRY_KEYS = ("bad gateway", "502", "503", "reset", "timed out", "connection reset", "unexpected end of json")


def _raw(args, body=None, retries=5):
    for attempt in range(1, retries + 1):
        cmd = ["gh", "api", *args]
        tmp = None
        if body is not None:
            tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
            json.dump(body, tmp)
            tmp.close()
            cmd += ["--input", tmp.name]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        finally:
            if tmp:
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass
        out = r.stdout
        if r.returncode == 0:
            return r.returncode, out
        blob = (r.stderr + out).lower()
        if any(k in blob for k in RETRY_KEYS) and attempt < retries:
            time.sleep(min(2 ** attempt, 8))
            continue
        return r.returncode, out
    return 1, ""


def _json(args, body=None):
    rc, out = _raw(args, body)
    if rc != 0:
        raise RuntimeError(f"gh api 失败 {args}: {out}")
    return json.loads(out) if out.strip() else {}


def collect_files():
    files = []
    for p in sorted(REPO_DIR.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(REPO_DIR)
        parts = set(rel.parts)
        if parts & EXCLUDE_DIRS:
            continue
        if p.suffix in EXCLUDE_EXT:
            continue
        files.append(p)
    return files


def bootstrap_empty():
    """空仓库: 先用 Contents API PUT 一个文件建初始 commit, 返回 (base_commit, base_tree)。"""
    readme = (REPO_DIR / "README.md").read_bytes()
    b64 = base64.b64encode(readme).decode("ascii")
    body = {"message": "OODForge init", "content": b64, "branch": "main"}
    res = _json(["repos/" + REPO + "/contents/README.md", "-X", "PUT"], body)
    sha = res["commit"]["sha"]
    cinfo = _json(["repos/" + REPO + "/commits/" + sha])
    print("[push] 空仓库已 bootstrap (Contents API PUT)")
    return sha, cinfo["commit"]["tree"]["sha"]


def main():
    import sys

    sys.stdout.reconfigure(line_buffering=True)
    # 0. 创建仓库 (失败可能因已存在, 忽略)
    subprocess.run(
        ["gh", "repo", "create", REPO, "--public", "--description", DESCRIPTION],
        capture_output=True, text=True,
    )
    time.sleep(2)

    files = collect_files()
    print(f"[push] 收集到 {len(files)} 个文件")

    # 1. 判断空仓库 (空仓库无法直接建 blob, 需先 bootstrap)
    rc, out = _raw(["repos/" + REPO + "/git/refs/heads/main"])
    if rc == 0:
        base_commit = json.loads(out)["object"]["sha"]
        cinfo = _json(["repos/" + REPO + "/commits/" + base_commit])
        base_tree = cinfo["commit"]["tree"]["sha"]
        print("[push] 非空仓库, 在现有 main 之上追加")
    else:
        base_commit, base_tree = bootstrap_empty()

    # 2. 建 blob (按内容去重)
    blob_cache = {}
    tree_entries = []
    total = len(files)
    for i, p in enumerate(files, 1):
        data = p.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        sha = blob_cache.get(b64)
        if sha is None:
            res = _json(["repos/" + REPO + "/git/blobs", "-X", "POST"], {"content": b64, "encoding": "base64"})
            sha = res["sha"]
            blob_cache[b64] = sha
        rel = p.relative_to(REPO_DIR).as_posix()
        tree_entries.append({"path": rel, "mode": "100644", "type": "blob", "sha": sha})
        if i % 25 == 0:
            print(f"[push] blob {i}/{total}")

    # 3. 建 tree
    tree_body = {"tree": tree_entries, "base_tree": base_tree}
    tree_sha = _json(["repos/" + REPO + "/git/trees", "-X", "POST"], tree_body)["sha"]

    # 4. 建 commit (嵌入作者 晨星)
    date = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    commit_body = {
        "message": "OODForge v0.1.0 · 分布外检测与概率校准系统 (CCOR 自适应路由)\n\n作者: 晨星 (CJX0712)",
        "tree": tree_sha,
        "parents": [base_commit],
        "author": {**AUTHOR, "date": date},
        "committer": {**AUTHOR, "date": date},
    }
    commit_sha = _json(["repos/" + REPO + "/git/commits", "-X", "POST"], commit_body)["sha"]
    print(f"[push] commit={commit_sha[:12]}")

    # 5. 更新 ref (PATCH; 若 ref 缺失则 POST)
    rc2, _ = _raw(["repos/" + REPO + "/git/refs/heads/main", "-X", "PATCH"], {"sha": commit_sha})
    if rc2 != 0:
        _json(["repos/" + REPO + "/git/refs", "-X", "POST"], {"ref": "refs/heads/main", "sha": commit_sha})
    print("[push] ref 已更新")

    # 6. 验证
    view = _json(["repos/" + REPO])
    print(f"[push] 仓库: {view.get('html_url')}")
    print(f"[push] 默认分支: {view.get('default_branch')}, 最新提交: {commit_sha[:12]}")


if __name__ == "__main__":
    main()
