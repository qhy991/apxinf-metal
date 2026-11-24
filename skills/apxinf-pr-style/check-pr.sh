#!/usr/bin/env bash
# Audit a branch for the patterns apxinf-pr-style forbids.
#
#   ./check-pr.sh [base]        # default base: origin/main
#   ./check-pr.sh HEAD~3        # audit the last three commits
#
# Exits 1 when it finds something; prints file:line for every hit.
# It is a text filter, not a reviewer: a clean run does not mean the PR is good.
set -uo pipefail

BASE=${1:-origin/main}
DIFF_ARGS=("$BASE...HEAD")

root=$(git rev-parse --show-toplevel 2>/dev/null) || { echo "not a git checkout" >&2; exit 2; }
cd "$root" || exit 2

status=0
note() { printf '%s\n' "$*"; }
fail() { printf 'FAIL  %s\n' "$*"; status=1; }

# Added lines only, without the leading '+', with their file and new line number.
added=$(git diff "${DIFF_ARGS[@]}" --unified=0 2>/dev/null | awk '
    /^\+\+\+ b\//  { file = substr($0, 7); next }
    /^@@/          { match($0, /\+[0-9]+/); line = substr($0, RSTART + 1, RLENGTH - 1) + 0; next }
    /^\+/          { printf "%s:%d:%s\n", file, line, substr($0, 2); line++ }
')

if [ -z "$added" ]; then
    note "no added lines against $BASE"
else
    # 1. Experiment narration and private-consumer coupling.
    while IFS= read -r hit; do
        case "$hit" in
            *"//"*|*"//!"*|*"/*"*|*"#"*)
                fail "experiment/private wording: $hit" ;;
        esac
    done < <(printf '%s\n' "$added" | grep -iE '^[^:]+:[0-9]+:.*(//|/\*|#).*\b(experiment|trial|benchmark run|first measured|we tried|demo|mario|prototype|scratch branch)\b' || true)

    # 2. Local paths and hostnames in any added line.
    while IFS= read -r hit; do
        fail "local path or hostname: $hit"
    done < <(printf '%s\n' "$added" | grep -E '/Users/|/home/[a-z]+/|\.local\b|mac-mini|MacBook|100\.[0-9]+\.[0-9]+\.[0-9]+' || true)

    # 3. Unfinished markers and commented-out code.
    while IFS= read -r hit; do
        fail "unfinished marker: $hit"
    done < <(printf '%s\n' "$added" | grep -E '\b(TODO|FIXME|XXX|TEMP|HACK)\b' || true)

    # 4. Dates in comments (paraphrase: a dated claim rots). Copyright lines are
    #    the exception and are matched out before the test.
    while IFS= read -r hit; do
        fail "dated comment: $hit"
    done < <(printf '%s\n' "$added" | grep -E '(//|/\*|#).*\b20[0-9]{2}-[01][0-9]-[0-3][0-9]\b' | grep -vi 'copyright' || true)

    # 5. Non-ASCII in rust/metal/C sources (the upstream tree is English-only).
    while IFS= read -r hit; do
        fail "non-ASCII source line: $hit"
    done < <(printf '%s\n' "$added" | grep -E '\.(rs|metal|mm|cpp|h|toml):' | LC_ALL=C grep -n '[^ -~]' | sed 's/^[0-9]*://' || true)
fi

# 6. Commit subjects.
subjects=$(git log --format='%s' "${DIFF_ARGS[@]}" 2>/dev/null)
if [ -n "$subjects" ]; then
    while IFS= read -r subject; do
        [ -z "$subject" ] && continue
        if ! printf '%s' "$subject" | grep -qE '^(feat|fix|perf|refactor|docs|test|build|ci|chore)(\([a-z0-9_,-]+\))?: .+'; then
            fail "commit subject lacks type(scope): $subject"
        fi
        if [ "${#subject}" -gt 72 ]; then
            fail "commit subject longer than 72 chars ($(printf '%s' "$subject" | wc -c | tr -d ' ')): $subject"
        fi
    done <<< "$subjects"
fi

# 7. Measurement claims belong in the PR description, not a commit body.
while IFS= read -r hit; do
    fail "measurement claim in commit body: $hit"
done < <(git log --format='%b' "${DIFF_ARGS[@]}" 2>/dev/null | grep -iE '\b(verified on|passed,|benchmark(ed)? on|ms on|seconds on)\b' || true)

if [ "$status" -eq 0 ]; then
    note "clean against $BASE"
fi
exit "$status"
