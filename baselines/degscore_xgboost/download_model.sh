#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")" && pwd)"
revision=b95af835cc38bd7dc55ed6e7397d124787f86996
model=model_files/bt_xgb/bt_xgb.model
sha=2a7cd3924e540298d8baac781992322e86ad4bde925a8f8741941a8009fce59c
destination="$root/checkpoint/bt_xgb.model"

verify() {
    printf '%s  %s\n' "$sha" "$1" | shasum -a 256 -c -
}

if [[ -f "$destination" ]] && verify "$destination"; then
    exit 0
fi

git lfs version >/dev/null
mkdir -p "$root/checkpoint"
staging="$(mktemp -d "$root/checkpoint/.download.XXXXXXXX")"
trap 'rm -rf -- "$staging"' EXIT
GIT_LFS_SKIP_SMUDGE=1 git clone --quiet --no-checkout https://github.com/eternagame/KaggleOpenVaccine.git "$staging/repo"
GIT_LFS_SKIP_SMUDGE=1 git -C "$staging/repo" checkout --quiet --detach "$revision"
git -C "$staging/repo" lfs install --local

git -C "$staging/repo" lfs pull --include="$model" --exclude=""
verify "$staging/repo/$model"
mv "$staging/repo/$model" "$destination"
