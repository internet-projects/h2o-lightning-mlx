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
SKIP = ["README.md", "test_*.py", "card_common.py", "__pycache__/*", ".cache/*", "*/.cache/*", "*.pyc", ".DS_Store", "._*"]
COMMIT_MSG = "H2O-Lightning-4B v1.3.1: two-way reading (previous build: v1.2.3 tag)"


def sh(*cmd, check=True):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if check and r.returncode:
        sys.exit(f"failed: {' '.join(cmd)}\n{r.stdout}{r.stderr}")
    return r.stdout.strip()


def fetch():
    sh("git", "fetch", "-q", "origin", "+refs/heads/main:refs/remotes/origin/main",
       f"+refs/heads/{BRANCH}:refs/remotes/origin/{BRANCH}", "--tags")


def remote_tags():
    out = set()
    for line in sh("git", "ls-remote", "--tags", "origin").splitlines():
        ref = line.split("\t")[-1]
        if ref.startswith("refs/tags/"):
            out.add(ref[len("refs/tags/"):].removesuffix("^{}"))
    return out


def git_tag(name, sha, msg):
    """Create the annotated tag locally unless it already exists there (a crash between tag and push), then push."""
    if subprocess.run(["git", "rev-parse", "-q", "--verify", f"refs/tags/{name}"], capture_output=True).returncode:
        sh("git", "tag", "-a", name, sha, "-m", msg)
    elif sh("git", "rev-parse", f"{name}^{{commit}}") != sha:
        sys.exit(f"local tag {name} points elsewhere; delete it (git tag -d {name}) and re-run")
    sh("git", "push", "origin", f"refs/tags/{name}")


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
    fetch()
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

    gh_tags = remote_tags()
    plan = [f"GitHub: tag {OLD} at main {main_sha[:7]}" + (" (exists)" if OLD in gh_tags else "")]
    for repo, _, _ in BUILDS:
        tags = [t.name for t in api.list_repo_refs(repo).tags]
        if REV in tags and OLD not in tags:
            sys.exit(f"{repo} has {REV} but no {OLD} tag: tag the previous build by hand first (never at main)")
        cur = api.model_info(repo).sha
        plan.append(f"HF {repo}: tag {OLD} at {cur[:7]}" + (" (exists)" if OLD in tags else ""))
        plan.append(f"HF {repo}: upload {REV} + card, tag {REV}" + (" (exists)" if REV in tags else ""))
    plan.append(f"GitHub: main {main_sha[:7]} -> {head[:7]} (fast-forward), tag {REV}"
                + (" (exists)" if REV in gh_tags else ""))
    print("\nplan:\n  " + "\n  ".join(plan))
    if not go:
        print("\ndry run: nothing changed. Run again with --go to publish.")
        return

    # ---- 1. GitHub: keep the previous release reachable
    if OLD not in gh_tags:
        git_tag(OLD, main_sha, "H2O-Lightning-4B v1.2.3 MLX port (one-way reading)")
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
        head_commit = api.list_repo_commits(repo)[0]
        if head_commit.title == COMMIT_MSG:          # uploaded by an earlier run that stopped before tagging
            oid = head_commit.commit_id
            print(f"HF {repo}: upload {oid[:7]} already there")
        else:
            ops = [CommitOperationAdd(path_in_repo=f, path_or_fileobj=os.path.join(folder, f)) for f in files_of(folder)]
            ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card))
            print(f"HF {repo}: uploading {len(ops)} files ...", flush=True)
            oid = api.create_commit(repo, operations=ops, commit_message=COMMIT_MSG).oid
        api.create_tag(repo, tag=REV, revision=oid, tag_message="H2O-Lightning-4B v1.3.1 (two-way reading)")
        print(f"HF {repo}: published {oid[:7]}, tagged {REV}")

    # ---- 3. GitHub: release the code
    fetch()
    main_now = sh("git", "rev-parse", "origin/main")
    if sh("git", "merge-base", main_now, head) != main_now:
        sys.exit("main moved during the upload; Hugging Face is already at v1.3.1. Merge main into the branch, push, re-run.")
    if main_now != head:
        sh("git", "push", "origin", f"{head}:refs/heads/main")
    if REV not in remote_tags():
        git_tag(REV, head, "H2O-Lightning-4B v1.3.1 MLX port (two-way reading)")
    print(f"GitHub: main is {head[:7]}, tagged {REV}")
    if subprocess.run(["which", "gh"], capture_output=True).returncode == 0:
        for tag, title, body in [
            (OLD, "v1.2.3 — H2O-Lightning-4B v1.2.3 (one-way)", "Previous release. Weights: `--revision v1.2.3` on Hugging Face."),
            (REV, "v1.3.1 — H2O-Lightning-4B v1.3.1 with two-way reading",
             "Two-way reading (H2O v1.3), served from one engine. JevBench public: 8-bit 203 / 231 two-way, "
             "bf16 205 / 231. See the README.")]:
            repo_gh = ["-R", "internet-projects/h2o-lightning-mlx"]
            if subprocess.run(["gh", "release", "view", tag] + repo_gh, capture_output=True).returncode == 0:
                print(f"GitHub: release {tag} exists")
                continue
            r = subprocess.run(["gh", "release", "create", tag] + repo_gh + ["--title", title, "--notes", body]
                               + (["--latest"] if tag == REV else ["--latest=false"]), capture_output=True, text=True)
            print(f"GitHub: release {tag} " + ("created" if r.returncode == 0 else f"FAILED: {r.stderr.strip()}"))
    else:
        print("GitHub: no gh CLI; create the v1.2.3 and v1.3.1 releases on github.com if you want them")
    print("\ndone.")


if __name__ == "__main__":
    main()
