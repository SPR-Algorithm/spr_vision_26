#!/usr/bin/env bash
# 将 wiki/ 目录下的 Markdown 同步到 GitHub Wiki 仓库。
# 用法：在已配置 GitHub 凭据的环境中执行 ./wiki/sync-wiki.sh

set -euo pipefail

REPO="SPR-Algorithm/spr_vision_26"
WIKI_URL="https://github.com/${REPO}.wiki.git"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP_DIR="$(mktemp -d)"

cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

echo "克隆 Wiki 仓库..."
git clone "$WIKI_URL" "$TMP_DIR/wiki"

echo "复制 Wiki 页面..."
cp "$SCRIPT_DIR"/Home.md "$TMP_DIR/wiki/"
cp "$SCRIPT_DIR"/调试工具.md "$TMP_DIR/wiki/"
cp "$SCRIPT_DIR"/配置指南.md "$TMP_DIR/wiki/"

cd "$TMP_DIR/wiki"
git add -A
if git diff --cached --quiet; then
  echo "Wiki 内容无变化，跳过推送。"
  exit 0
fi

git commit -m "docs: sync wiki pages from main repo"
git push origin master

echo "Wiki 同步完成: https://github.com/${REPO}/wiki"
