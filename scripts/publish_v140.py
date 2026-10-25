#!/usr/bin/env python3
"""Publish the H2O v1.4.0 MLX update (run on the Mac from ~/local-llms/h2o/v1.4.0/repo, with your HF login and
git push access to internet-projects/h2o-lightning-mlx).

    .venv/bin/python scripts/publish_v140.py          # dry run: checks everything, prints the plan, changes nothing
    .venv/bin/python scripts/publish_v140.py --go     # do it

Order (each step is skipped when already done, so a re-run resumes):
  1. GitHub: push the branch v1.4.0 (this checkout was cloned from the local v1.3.1 checkout, so the GitHub remote is
     added as `github`); the previous release is already tagged v1.3.1 there.
  2. Hugging Face, for each of netprojects/h2o-lightning-4b-mlx-q8 and -q4: make sure the current main is tagged
     v1.3.1 (it is, from the 19 Oct release), upload the v1.4.0 build with its new card as ONE commit, tag it v1.4.0.
  3. GitHub: fast-forward main to v1.4.0, tag v1.4.0, push, and create the GitHub release.
The previous builds stay reachable at the v1.3.1 and v1.2.3 tags.
"""
import argparse
import fnmatch
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(HERE)
REV, OLD, BRANCH = "v1.4.0", "v1.3.1", "v1.4.0"
REMOTE_BRANCH = "v1.4.0-update"   # not "v1.4.0": that name is the release tag
H2O_COMMIT = "d42875078bafb155af14d609d85071091418e83f"     # h2oai/h2o-lightning-4b tag v1.4.0
REMOTE, REMOTE_URL = "github", "https://github.com/internet-projects/h2o-lightning-mlx"
BUILDS = [("netprojects/h2o-lightning-4b-mlx-q8", "mlx-q8", "hf_cards/q8.md"),
          ("netprojects/h2o-lightning-4b-mlx-q4", "mlx-q4", "hf_cards/q4.md")]
# shipped with each build: weights, configs, tokenizer, H2O's shim with the modules it loads, label rows, the engine
# checksum file and the receipt example; not H2O's test files or local caches
SKIP = ["README.md", "test_*.py", "pytest.ini", "card_common.py", "label_rows.lightning", "engine/*", "card_report.json", "card_snapshot_*", "card_snapshot_*/*", "__pycache__/*", ".cache/*", "*/.cache/*", "*.pyc", ".DS_Store", "._*",
        "recipe.json", ".pytest_cache/*", "*/.pytest_cache/*"]
COMMIT_MSG = "H2O-Lightning-4B v1.4.0 (previous build: v1.3.1 tag)"


def sh(*cmd, check=True):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if check and r.returncode:
        sys.exit(f"failed: {' '.join(cmd)}\n{r.stdout}{r.stderr}")
    return r.stdout.strip()


def ensure_remote():
    if REMOTE not in sh("git", "remote").split():
        sh("git", "remote", "add", REMOTE, REMOTE_URL)
    sh("git", "fetch", "-q", REMOTE, "+refs/heads/*:refs/remotes/github/*", "--tags")


def remote_tags():
    out = set()
    for line in sh("git", "ls-remote", "--tags", REMOTE).splitlines():
        ref = line.split("\t")[-1]
        if ref.startswith("refs/tags/"):
            out.add(ref[len("refs/tags/"):].removesuffix("^{}"))
    return out


def git_tag(name, sha, msg):
    if subprocess.run(["git", "rev-parse", "-q", "--verify", f"refs/tags/{name}"], capture_output=True).returncode:
        sh("git", "tag", "-a", name, sha, "-m", msg)
    elif sh("git", "rev-parse", f"{name}^{{commit}}") != sha:
        sys.exit(f"local tag {name} points elsewhere; delete it (git tag -d {name}) and re-run")
    sh("git", "push", REMOTE, f"refs/tags/{name}")


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
    tok = (who.get("auth") or {}).get("accessToken") or {}
    if tok.get("role") == "read":
        sys.exit("this HF token is read-only: make a Write token at https://huggingface.co/settings/tokens, then: .venv/bin/hf auth login")
    print(f"HF token: {tok.get('role') or 'unknown role'}")
    ensure_remote()
    if sh("git", "rev-parse", "--abbrev-ref", "HEAD") != BRANCH:
        sys.exit(f"check out {BRANCH} first")
    if sh("git", "status", "--porcelain", "--untracked-files=no"):
        sys.exit("uncommitted changes in the repo")
    head = sh("git", "rev-parse", "HEAD")
    main_sha = sh("git", "rev-parse", "github/main")
    if sh("git", "merge-base", main_sha, head) != main_sha:
        sys.exit("GitHub main has moved on since this branch was made; merge it first")
    for repo, folder, card in BUILDS:
        p = os.path.join(folder, ".h2o_revision")
        stamp = open(p).read().strip() if os.path.exists(p) else None
        if stamp != H2O_COMMIT:
            sys.exit(f"{folder} is not a {REV} build (stamp {stamp})")
        if not os.path.exists(card):
            sys.exit(f"missing {card}")
        fs = files_of(folder)
        for need in ("config.json", "model.safetensors", "h2o_lightning_shim.py", "serve_config.json", "box.py", ".h2o_revision"):
            if need not in fs:
                sys.exit(f"{folder} has no {need}")
        size = sum(os.path.getsize(os.path.join(folder, f)) for f in fs) / 1e9
        print(f"{repo}: upload {len(fs)} files from {folder}/ ({size:.1f} GB) + {card} as README.md")

    gh_tags = remote_tags()
    pushed = subprocess.run(["git", "rev-parse", "-q", "--verify", f"github/{REMOTE_BRANCH}"], capture_output=True, text=True).stdout.strip()
    plan = [f"GitHub: push branch {REMOTE_BRANCH} ({head[:7]})" + (" (already pushed)" if pushed == head else ""),
            f"GitHub: {OLD} tag " + ("present" if OLD in gh_tags else "MISSING (will tag current main)")]
    for repo, _, _ in BUILDS:
        tags = [t.name for t in api.list_repo_refs(repo).tags]
        if REV in tags and OLD not in tags:
            sys.exit(f"{repo} has {REV} but no {OLD} tag: tag the previous build by hand first (never at main)")
        cur = api.model_info(repo).sha
        plan.append(f"HF {repo}: {OLD} tag " + ("present" if OLD in tags else f"to add at main {cur[:7]}"))
        plan.append(f"HF {repo}: upload {REV} + card, tag {REV}" + (" (exists)" if REV in tags else ""))
    plan.append(f"GitHub: main {main_sha[:7]} -> {head[:7]} (fast-forward), tag {REV}, release" + (" (tag exists)" if REV in gh_tags else ""))
    print("\nplan:\n  " + "\n  ".join(plan))
    if not go:
        print("\ndry run: nothing changed. Run again with --go to publish.")
        return

    # ---- 1. GitHub branch
    if pushed != head:
        sh("git", "push", REMOTE, f"{head}:refs/heads/{REMOTE_BRANCH}")
        print(f"GitHub: pushed {REMOTE_BRANCH}")
    if OLD not in gh_tags:
        git_tag(OLD, main_sha, "H2O-Lightning-4B v1.3.1 MLX port (two-way reading)")

    # ---- 2. Hugging Face
    for repo, folder, card in BUILDS:
        tags = [t.name for t in api.list_repo_refs(repo).tags]
        if OLD not in tags:
            api.create_tag(repo, tag=OLD, revision=api.model_info(repo).sha, tag_message="H2O-Lightning-4B v1.3.1")
            print(f"HF {repo}: tagged {OLD}")
        if REV in tags:
            print(f"HF {repo}: {REV} already published")
            continue
        head_commit = api.list_repo_commits(repo)[0]
        if head_commit.title == COMMIT_MSG:
            oid = head_commit.commit_id
            print(f"HF {repo}: upload {oid[:7]} already there")
        else:
            ops = [CommitOperationAdd(path_in_repo=f, path_or_fileobj=os.path.join(folder, f)) for f in files_of(folder)]
            ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card))
            print(f"HF {repo}: uploading {len(ops)} files ...", flush=True)
            oid = api.create_commit(repo, operations=ops, commit_message=COMMIT_MSG).oid
        api.create_tag(repo, tag=REV, revision=oid, tag_message="H2O-Lightning-4B v1.4.0")
        print(f"HF {repo}: published {oid[:7]}, tagged {REV}")

    # ---- 3. GitHub: release the code
    ensure_remote()
    main_now = sh("git", "rev-parse", "github/main")
    if sh("git", "merge-base", main_now, head) != main_now:
        sys.exit("main moved during the upload; Hugging Face is already at v1.4.0. Merge main into the branch, push, re-run.")
    if main_now != head:
        sh("git", "push", REMOTE, f"{head}:refs/heads/main")
    if REV not in remote_tags():
        git_tag(REV, head, "H2O-Lightning-4B v1.4.0 MLX port (guided Box generation)")
    print(f"GitHub: main is {head[:7]}, tagged {REV}")
    if subprocess.run(["which", "gh"], capture_output=True).returncode == 0:
        repo_gh = ["-R", "internet-projects/h2o-lightning-mlx"]
        if subprocess.run(["gh", "release", "view", REV] + repo_gh, capture_output=True).returncode == 0:
            print(f"GitHub: release {REV} exists")
        else:
            body = ("H2O-Lightning-4B v1.4.0 on MLX. JevBench public: 8-bit 206 / 231 two-way, the same as H2O's own "
                    "vLLM result; Box questions answered by guided generation. See the README.")
            r = subprocess.run(["gh", "release", "create", REV] + repo_gh + ["--title", "v1.4.0 — H2O-Lightning-4B v1.4.0",
                               "--notes", body, "--latest"], capture_output=True, text=True)
            print(f"GitHub: release {REV} " + ("created" if r.returncode == 0 else f"FAILED: {r.stderr.strip()}"))
    else:
        print("GitHub: no gh CLI; create the v1.4.0 release on github.com if you want one")
    print("\ndone.")


if __name__ == "__main__":
    main()
