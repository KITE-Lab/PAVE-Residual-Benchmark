#!/usr/bin/env bash
set -euo pipefail

REMOTE_URL="git@github.com:KITE-Lab/PAVE-Residual-Benchmark.git"
DEFAULT_BRANCH="main"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/git_submit_current.sh [--dry-run] [commit message]

Examples:
  ./scripts/git_submit_current.sh
  ./scripts/git_submit_current.sh "update residual evaluation"
  ./scripts/git_submit_current.sh --dry-run
EOF
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
DRY_RUN=0

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

COMMIT_MESSAGE="${*:-}"
if [[ -z "${COMMIT_MESSAGE}" ]]; then
  COMMIT_MESSAGE="chore: update PAVE-Residual-Benchmark $(date '+%Y-%m-%d %H:%M:%S')"
fi

cd "${PROJECT_ROOT}"

echo "[INFO] repository root: ${PROJECT_ROOT}"
echo "[INFO] remote: ${REMOTE_URL}"
echo

IS_GIT_REPO=0
if git -C "${PROJECT_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  ACTUAL_ROOT="$(git -C "${PROJECT_ROOT}" rev-parse --show-toplevel)"
  [[ "${ACTUAL_ROOT}" == "${PROJECT_ROOT}" ]] || die "project is inside another Git repository: ${ACTUAL_ROOT}"
  IS_GIT_REPO=1
fi

if [[ "${IS_GIT_REPO}" -eq 0 ]]; then
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[DRY-RUN] project is not a Git repository yet"
    echo "[DRY-RUN] would run: git init"
    echo "[DRY-RUN] would run: git checkout -B ${DEFAULT_BRANCH}"
    echo "[DRY-RUN] would run: git remote add origin ${REMOTE_URL}"
    echo "[DRY-RUN] commit message: ${COMMIT_MESSAGE}"
    echo "[DRY-RUN] would stage non-ignored project files and push to origin/${DEFAULT_BRANCH}"
    exit 0
  fi

  echo "[INFO] initializing Git repository"
  git init
  git checkout -B "${DEFAULT_BRANCH}"
  git remote add origin "${REMOTE_URL}"
fi

GIT_DIR="$(git -C "${PROJECT_ROOT}" rev-parse --git-dir)"
BRANCH="$(git symbolic-ref --quiet --short HEAD 2>/dev/null)" || die "refusing to submit from detached HEAD"
[[ "${BRANCH}" == "${DEFAULT_BRANCH}" ]] || die "expected branch '${DEFAULT_BRANCH}', got '${BRANCH}'"

if git remote get-url origin >/dev/null 2>&1; then
  ORIGIN_URL="$(git remote get-url origin)"
  [[ "${ORIGIN_URL}" == "${REMOTE_URL}" ]] || die "origin points to '${ORIGIN_URL}', expected '${REMOTE_URL}'"
else
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[DRY-RUN] would run: git remote add origin ${REMOTE_URL}"
  else
    git remote add origin "${REMOTE_URL}"
  fi
fi

if git rev-parse -q --verify MERGE_HEAD >/dev/null 2>&1; then
  die "merge is in progress; resolve it before submitting"
fi

if [[ -d "${GIT_DIR}/rebase-merge" || -d "${GIT_DIR}/rebase-apply" ]]; then
  die "rebase is in progress; resolve it before submitting"
fi

if [[ -n "$(git diff --name-only --diff-filter=U)" ]]; then
  die "unmerged files are present; resolve conflicts before submitting"
fi

if [[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]]; then
  echo "[INFO] nothing to commit"
  exit 0
fi

echo "[INFO] current status:"
git status --short --untracked-files=all
echo

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "[DRY-RUN] commit message: ${COMMIT_MESSAGE}"
  echo "[DRY-RUN] files that would be staged by git add -A:"
  git add -A --dry-run
  echo "[DRY-RUN] would run: git commit -m \"${COMMIT_MESSAGE}\""
  echo "[DRY-RUN] would check whether origin/${DEFAULT_BRANCH} exists"
  echo "[DRY-RUN] would run: git pull --rebase --autostash origin ${DEFAULT_BRANCH} if remote branch exists"
  echo "[DRY-RUN] would run: git push -u origin ${DEFAULT_BRANCH}"
  exit 0
fi

echo "[INFO] staging all non-ignored changes"
git add -A

if git diff --cached --quiet; then
  echo "[INFO] no staged changes after git add -A"
  exit 0
fi

echo "[INFO] committing"
git commit -m "${COMMIT_MESSAGE}"

if git ls-remote --exit-code --heads origin "${DEFAULT_BRANCH}" >/dev/null 2>&1; then
  echo "[INFO] rebasing against origin/${DEFAULT_BRANCH}"
  git pull --rebase --autostash origin "${DEFAULT_BRANCH}"
else
  echo "[INFO] remote branch origin/${DEFAULT_BRANCH} not found or unavailable; skipping pull"
fi

echo "[INFO] pushing to origin/${DEFAULT_BRANCH}"
git push -u origin "${DEFAULT_BRANCH}"

echo "[INFO] submitted successfully"
