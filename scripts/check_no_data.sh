#!/usr/bin/env bash
# Stops if anything under data/ is staged, tracked, or in Git history.
# Used as the pre-commit hook and in CI (DATA_CONTRACT.md section 3.2).
#
# Install the hook once with:
#   ln -sf ../../scripts/check_no_data.sh .git/hooks/pre-commit

cd "$(git rev-parse --show-toplevel)" || exit 1

staged=$(git diff --cached --name-only | grep -c '^data/')
tracked=$(git ls-files -- data/ | wc -l | tr -d ' ')
history=$(git log --all --name-only --pretty=format: -- data/ | grep -c '^data/')

if [ "$staged" -ne 0 ] || [ "$tracked" -ne 0 ] || [ "$history" -ne 0 ]; then
    echo "E_GOV_RESTRICTED_LEAK: data/ paths found (staged=$staged tracked=$tracked history=$history)." >&2
    echo "Do not commit. If this reached Git history, follow DATA_CONTRACT.md section 4.4." >&2
    exit 4
fi

echo "check_no_data: OK (no data/ paths staged, tracked, or in history)"
exit 0
