#!/usr/bin/env bash
# Validates commit subject lines against Conventional Commits
# (https://www.conventionalcommits.org). Used by both .githooks/commit-msg
# (single message) and CI (a range of commits) so the rule can't drift
# between local and CI enforcement.
#
# Usage:
#   check-commit-msg.sh <<< "$subject_line"
#   git log --format=%s <range> | check-commit-msg.sh

set -euo pipefail

TYPES='feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert'
PATTERN="^(${TYPES})(\([a-z0-9_-]+\))?!?: .+"
MERGE_PATTERN="^Merge "

failed=0
while IFS= read -r subject; do
  [ -z "$subject" ] && continue
  if [[ "$subject" =~ $MERGE_PATTERN ]]; then
    continue
  fi
  if [[ ! "$subject" =~ $PATTERN ]]; then
    echo "Not a Conventional Commit: \"$subject\"" >&2
    echo "  expected: <${TYPES}>(<scope>)?!?: <description>" >&2
    failed=1
  fi
done

exit "$failed"
