#!/usr/bin/env bash
set -euo pipefail

readonly WORK_EXCLUDE_PATHSPEC=':(exclude).work'
readonly ISSUE_FORGE_EXCLUDE_PATHSPEC=':(exclude)vendor/issue_forge'

fail() {
  printf '%s\n' "$1" >&2
  exit 1
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    fail "Missing required check command: $1"
  fi
}

collect_changed_files() {
  local base_ref="$1"

  {
    git diff --name-only "$base_ref" -- . "$WORK_EXCLUDE_PATHSPEC" "$ISSUE_FORGE_EXCLUDE_PATHSPEC"
    git diff --name-only --cached -- . "$WORK_EXCLUDE_PATHSPEC" "$ISSUE_FORGE_EXCLUDE_PATHSPEC"
    git diff --name-only -- . "$WORK_EXCLUDE_PATHSPEC" "$ISSUE_FORGE_EXCLUDE_PATHSPEC"
    git ls-files --others --exclude-standard -- . "$WORK_EXCLUDE_PATHSPEC" "$ISSUE_FORGE_EXCLUDE_PATHSPEC"
  } | awk 'NF' | LC_ALL=C sort -u
}

main() {
  local base_ref
  local path
  local run_python=0
  local run_frontend_build=0
  local run_doctor=0
  local -a changed_files=()
  local -a shell_targets=()

  if [[ "$#" -ne 1 ]]; then
    fail "Usage: $0 <base_ref>"
  fi

  base_ref="$1"

  require_command git
  require_command awk

  if ! git rev-parse --verify "$base_ref" >/dev/null 2>&1; then
    fail "Missing base ref for checks: $base_ref"
  fi

  mapfile -t changed_files < <(collect_changed_files "$base_ref")

  if [[ "${#changed_files[@]}" -eq 0 ]]; then
    printf 'No changes detected relative to %s\n' "$base_ref"
    return 0
  fi

  printf 'Changed files relative to %s:\n' "$base_ref"
  printf ' - %s\n' "${changed_files[@]}"

  for path in "${changed_files[@]}"; do
    case "$path" in
      *.sh)
        shell_targets+=("$path")
        ;;
    esac

    case "$path" in
      AGENTS.md|docs/*|.issue_forge/*)
        run_doctor=1
        ;;
    esac

    case "$path" in
      *.py|pyproject.toml|pytest.ini|.devcontainer/requirements-dev.txt)
        run_python=1
        ;;
      app/static/*|app/tests/e2e/*|playwright.config.ts|package.json)
        run_python=1
        ;;
      app/web/*|app/package.json|app/package-lock.json|app/vite.config.ts|app/vitest.config.mjs)
        run_python=1
        run_frontend_build=1
        ;;
    esac
  done

  if [[ "${#shell_targets[@]}" -gt 0 ]]; then
    require_command shellcheck
    printf 'shellcheck: %s target(s)\n' "${#shell_targets[@]}"
    shellcheck -x "${shell_targets[@]}"
  else
    printf 'shellcheck: skipped\n'
  fi

  if [[ "$run_doctor" -eq 1 ]]; then
    printf 'doctor: ./vendor/issue_forge/tools/codex/doctor.sh\n'
    ./vendor/issue_forge/tools/codex/doctor.sh
  else
    printf 'doctor: skipped\n'
  fi

  if [[ "$run_frontend_build" -eq 1 ]]; then
    require_command npm
    printf 'frontend build: cd app && npm run build\n'
    (
      cd app
      npm run build
    )
  else
    printf 'frontend build: skipped\n'
  fi

  if [[ "$run_python" -eq 1 ]]; then
    require_command python
    require_command ruff
    export PYTHONPATH="$PWD${PYTHONPATH:+:${PYTHONPATH}}"

    printf 'compileall: python -m compileall -q app\n'
    python -m compileall -q app

    printf 'ruff: ruff check app\n'
    ruff check app

    printf 'pytest: python -m pytest -q\n'
    python -m pytest -q
  else
    printf 'python checks: skipped\n'
  fi
}

main "$@"
