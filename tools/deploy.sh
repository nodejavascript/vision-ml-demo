#!/usr/bin/env bash
#
# Build, test, publish, purge, smoke-check, verify live.
#
# There is no auto-deploy on this repo: this script IS the deploy. It builds from TypeScript
# first, so a stale `site/` can never be published by accident, and it runs all three kinds
# of test — unit, end-to-end, live — because standard part 6 says a site whose tests have
# not been run is not deployed yet.
#
# The `no-store` header the site depends on lives in the Caddy block on the droplet, not
# here. End-to-end test 1 asserts it is really being sent, so a server change that drops it
# fails loudly rather than quietly making every deploy invisible to a returning visitor.

set -euo pipefail
cd "$(dirname "$0")/.."

echo "== build =="
npm run build

echo "== unit tests =="
node --test test/static.test.js

echo "== end-to-end tests =="
node --test test/e2e.test.js

echo "== publish =="
rsync -az --delete --rsync-path="sudo rsync" site/ dvs-sites:/srv/vision-ml-demo/
ssh dvs-sites 'sudo chmod -R a+rX /srv/vision-ml-demo'

echo "== purge the CDN =="
( cd ~/Documents/git/gitlab.com/datavisionstudios/docker-compose-master && .venv/bin/python3 ~/.cloudflare_purge.py --all )

echo "== smoke check =="
for path in / /app.js /consent.js /styles.css /manifest.webmanifest /robots.txt /sitemap.xml /og.png; do
  code=$(curl -s -o /dev/null -w '%{http_code}' "https://vision-ml-demo.nodejavascript.com$path")
  printf '  %-26s %s\n' "$path" "$code"
  [ "$code" = "200" ] || { echo "FAILED: $path returned $code"; exit 1; }
done

cache=$(curl -sI https://vision-ml-demo.nodejavascript.com/ | grep -i '^cache-control' | tr -d '\r')
case "$cache" in
  *no-store*) echo "  cache-control: $cache" ;;
  *) echo "FAILED: the shell is not no-store ($cache) — a deploy would be invisible to a returning visitor"; exit 1 ;;
esac

echo "== live check =="
node tools/verify-live-consent.mjs

echo
echo "deployed. The end-to-end suite can be re-run against a local server with: npm run test:e2e"
