#!/usr/bin/env python3
"""Publish the H2O v1.3.1 MLX update (run on the Mac, from the repo root, with your HF login and git push access).

    .venv/bin/python scripts/publish_v131.py          # dry run: checks everything, prints the plan, changes nothing
    .venv/bin/python scripts/publish_v131.py --go     # do it

Order (each step is skipped when already done, so a re-run resumes):
  1. GitHub: tag the current main as v1.2.3 (the previous release) and push the tag.
  2. Hugging Face, for each of netprojects/h2o-lightning-4b-mlx-q8 and -q4: tag the current main as v1.2.3, upload the
     v1.3.1 build (mlx-q8 / mlx-q4) with its new card (hf_cards/q8.md / q4.md) as ONE commit, then tag it v1.3.1.
  3. GitHub: fast-forward main to this branch (v1.3.1-two-way), tag it v1.3.1 and push.
The old builds stay reachable at the v1.2.3 tags (`--revision v1.2.3`, `git checkout v1.2.3`).
"""
import argparse
import fnmatch
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(HERE)
REV, OLD, BRANCH = "v1.3.1", "v1.2.3", "v1.3.1-two-way"
BUILDS = [("netprojects/h2o-lightning-4b-mlx-q8", "mlx-q8", "hf_cards/q8.md"),
          ("netprojects/h2o-lightning-4b-mlx-q4", "mlx-q4", "hf_cards/q4.md")]
# shipped with each build: the weights, configs, tokenizer, and H2O's shim with the modules it loads (as before);
# not H2O's test files or local caches
SKIP = ["README.md", "test_*.py", "card_common.py", "__pycache__/*", ".cache/*", "*.pyc", ".DS_Store"]


def sh(*cmd, check=True):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if check and r.returncode:
        sys.exit(f"failed: {' '.join(cmd)}\n{r.stdout}{r.stderr}")
    return r.stdout.strip()


def files_of(folder):
    out = []
    for root, dirs, names in os.walk(folder):
        for n in names:
            rel = os.path.relpath(os.path.join(root, n), folder)
            if not any(fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(n, p) for p in SKIP):
                out.append(rel)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true", help="actually publish (default: dry run)")
    go = ap.parse_args().go
    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi()

    # ---- checks
    who = api.whoami()
    orgs = [o.get("name") for o in who.get("orgs", [])]
    print(f"HF login: {who.get('name')} (orgs: {', '.join(orgs) or '-'})")
    if "netprojects" not in orgs and who.get("name") != "netprojects":
        sys.exit("this HF login cannot write to the netprojects org (hf auth login with a token that can)")
    sh("git", "fetch", "-q", "origin", "main", BRANCH, "--tags")
    if sh("git", "rev-parse", "--abbrev-ref", "HEAD") != BRANCH:
        sys.exit(f"check out {BRANCH} first")
    if sh("git", "status", "--porcelain", "--untracked-files=no"):
        sys.exit("uncommitted changes in the repo")
    head, remote = sh("git", "rev-parse", "HEAD"), sh("git", "rev-parse", f"origin/{BRANCH}")
    if head != remote:
        sys.exit(f"local {BRANCH} is not the pushed one: git pull")
    main_sha = sh("git", "rev-parse", "origin/main")
    if sh("git", "merge-base", main_sha, head) != main_sha:
        sys.exit("main has moved on since the branch was made; merge it first")
    for repo, folder, card in BUILDS:
        stamp = open(os.path.join(folder, ".h2o_revision")).read().strip() if os.path.exists(os.path.join(folder, ".h2o_revision")) else None
        if stamp != REV:
            sys.exit(f"{folder} is not a {REV} build (stamp {stamp})")
        if not os.path.exists(card):
            sys.exit(f"missing {card}")
        fs = files_of(folder)
        for need in ("config.json", "model.safetensors", "h2o_lightning_shim.py", "serve_config.json", ".h2o_revision"):
            if need not in fs:
                sys.exit(f"{folder} has no {need}")
        size = sum(os.path.getsize(os.path.join(folder, f)) for f in fs) / 1e9
        print(f"{repo}: upload {len(fs)} files from {folder}/ ({size:.1f} GB) + {card} as README.md")

    gh_tags = sh("git", "ls-remote", "--tags", "origin")
    plan = [f"GitHub: tag {OLD} at main {main_sha[:7]}" + (" (exists)" if f"refs/tags/{OLD}" in gh_tags else "")]
    for repo, _, _ in BUILDS:
        tags = [t.name for t in api.list_repo_refs(repo).tags]
        cur = api.model_info(repo).sha
        plan.append(f"HF {repo}: tag {OLD} at {cur[:7]}" + (" (exists)" if OLD in tags else ""))
        plan.append(f"HF {repo}: upload {REV} + card, tag {REV}" + (" (exists)" if REV in tags else ""))
    plan.append(f"GitHub: main {main_sha[:7]} -> {head[:7]} (fast-forward), tag {REV}"
                + (" (exists)" if f"refs/tags/{REV}" in gh_tags else ""))
    print("\nplan:\n  " + "\n  ".join(plan))
    if not go:
        print("\ndry run: nothing changed. Run again with --go to publish.")
        return

    # ---- 1. GitHub: keep the previous release reachable
    if f"refs/tags/{OLD}" not in gh_tags:
        sh("git", "tag", "-a", OLD, main_sha, "-m", "H2O-Lightning-4B v1.2.3 MLX port (one-way reading)")
        sh("git", "push", "origin", f"refs/tags/{OLD}")
        print(f"GitHub: tagged {OLD}")

    # ---- 2. Hugging Face
    for repo, folder, card in BUILDS:
        tags = [t.name for t in api.list_repo_refs(repo).tags]
        if OLD not in tags:
            api.create_tag(repo, tag=OLD, revision=api.model_info(repo).sha,
                           tag_message="H2O-Lightning-4B v1.2.3 (one-way reading)")
            print(f"HF {repo}: tagged {OLD}")
        if REV in tags:
            print(f"HF {repo}: {REV} already published")
            continue
        ops = [CommitOperationAdd(path_in_repo=f, path_or_fileobj=os.path.join(folder, f)) for f in files_of(folder)]
        ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card))
        print(f"HF {repo}: uploading {len(ops)} files ...", flush=True)
        info = api.create_commit(repo, operations=ops,
                                 commit_message="H2O-Lightning-4B v1.3.1: two-way reading (previous build: v1.2.3 tag)")
        api.create_tag(repo, tag=REV, revision=info.oid, tag_message="H2O-Lightning-4B v1.3.1 (two-way reading)")
        print(f"HF {repo}: published {info.oid[:7]}, tagged {REV}")

    # ---- 3. GitHub: release the code
    sh("git", "push", "origin", f"{head}:refs/heads/main")
    if f"refs/tags/{REV}" not in gh_tags:
        sh("git", "tag", "-a", REV, head, "-m", "H2O-Lightning-4B v1.3.1 MLX port (two-way reading)")
        sh("git", "push", "origin", f"refs/tags/{REV}")
    print(f"GitHub: main is {head[:7]}, tagged {REV}")
    if subprocess.run(["which", "gh"], capture_output=True).returncode == 0:
        for tag, title, body in [
            (OLD, "v1.2.3 — H2O-Lightning-4B v1.2.3 (one-way)", "Previous release. Weights: `--revision v1.2.3` on Hugging Face."),
            (REV, "v1.3.1 — H2O-Lightning-4B v1.3.1 with two-way reading",
             "Two-way reading (H2O v1.3), served from one engine. JevBench public: 8-bit 203 / 231 two-way, "
             "bf16 205 / 231. See the README.")]:
            subprocess.run(["gh", "release", "create", tag, "-R", "internet-projects/h2o-lightning-mlx",
                            "--title", title, "--notes", body] + (["--latest"] if tag == REV else ["--latest=false"]),
                           capture_output=True)
        print("GitHub: releases created")
    print("\ndone.")


if __name__ == "__main__":
    main()
