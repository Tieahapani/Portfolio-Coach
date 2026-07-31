#!/bin/bash
# Deploy Portfolio Coach: push local commits, pull on the droplet, restart, verify.
# Usage: ./deploy.sh
set -e

SERVER="root@68.183.172.77"
KEY="$HOME/.ssh/id_ed25519_personal"

# Load the SSH key if the agent doesn't have it yet
if ! ssh-add -l 2>/dev/null | grep -q "id_ed25519_personal"; then
    echo "→ Loading SSH key..."
    ssh-add "$KEY"
fi

# Warn about uncommitted changes (they won't deploy)
if ! git diff --quiet app/ 2>/dev/null; then
    echo "⚠️  You have uncommitted changes in app/ — they will NOT be deployed."
    read -p "Continue anyway? [y/N] " ans
    [[ "$ans" == "y" || "$ans" == "Y" ]] || exit 1
fi

echo "→ Pushing to GitHub..."
git push

echo "→ Deploying on droplet..."
ssh "$SERVER" "cd /opt/portfolio-coach && git pull && systemctl restart portfolio"

echo "→ Waiting for app to boot (ChromaDB is slow)..."
sleep 20

STATUS=$(ssh "$SERVER" "systemctl is-active portfolio")
HTTP=$(curl -s -o /dev/null -w '%{http_code}' https://portfolio-coach.duckdns.org)

if [[ "$STATUS" == "active" && "$HTTP" == "200" ]]; then
    echo "✅ Deployed — service $STATUS, site returns $HTTP"
else
    echo "❌ Something's wrong — service: $STATUS, site: $HTTP"
    echo "   Check logs: ssh $SERVER 'journalctl -u portfolio -n 50'"
    exit 1
fi
