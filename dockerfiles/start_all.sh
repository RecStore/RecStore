#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

bash start_docker.sh

until sudo docker inspect -f '{{.State.Running}}' xmh_recstore 2>/dev/null | grep -q true; do
  sleep 1
done

sudo docker exec -it xmh_recstore bash -c 'bash dockerfiles/codex/init_codex.sh; exec bash'
