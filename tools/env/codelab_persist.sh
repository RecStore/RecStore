#!/usr/bin/env bash

# Persistence path resolution is intentionally kept at the top of this file.
# Priority: --persist-dir, CODELAB_PERSIST_DIR, then Codelab auto-detection.
# Source this file to resolve and export available values silently.

# Site-specific mount layout. Override these variables when paths change.
CODELAB_SSD_USER_ROOT="${CODELAB_SSD_USER_ROOT:-/mnt/dolphinfs/ssd_pool/docker/user}"
CODELAB_HDD_USER_ROOT="${CODELAB_HDD_USER_ROOT:-/mnt/dolphinfs/hdd_pool/docker/user}"
CODELAB_PRIVATE_MOUNT_PREFIX="${CODELAB_PRIVATE_MOUNT_PREFIX:-/opt/meituan/dolphinfs_}"

die() {
  printf 'Error: %s\n' "$1" >&2
  return 1
}

valid_name() {
  [[ "$1" =~ ^[A-Za-z0-9._-]+$ ]]
}

detect_mis() {
  if [[ -n "${CODELAB_MIS:-}" ]]; then
    valid_name "$CODELAB_MIS" || {
      die "CODELAB_MIS has an invalid format."
      return 1
    }
    return
  fi

  command -v findmnt >/dev/null 2>&1 || {
    die "findmnt is required for automatic detection."
    return 1
  }

  local target source mount_source candidate
  while read -r target source; do
    [[ "$target" == "${CODELAB_PRIVATE_MOUNT_PREFIX}"* ]] || continue
    mount_source="${source#*[}"
    mount_source="${mount_source%]}"
    candidate="${mount_source##*/}"
    if valid_name "$candidate"; then
      CODELAB_MIS="$candidate"
      return
    fi
  done < <(findmnt -rn -t fuse.bgfuse -o TARGET,SOURCE 2>/dev/null)

  die "Unable to detect MIS; set CODELAB_MIS explicitly."
  return 1
}

detect_persist_dir() {
  if [[ -n "${CODELAB_PERSIST_DIR:-}" ]]; then
    [[ -d "$CODELAB_PERSIST_DIR" && -w "$CODELAB_PERSIST_DIR" ]] ||
      {
        die "CODELAB_PERSIST_DIR is not a writable directory."
        return 1
      }
    CODELAB_PERSIST_DIR="$(cd "$CODELAB_PERSIST_DIR" && pwd -P)" || return 1
    return
  fi

  local user_name candidate target source mount_source
  user_name="$(id -un)"

  for candidate in \
    "${CODELAB_SSD_USER_ROOT}/${user_name}/${CODELAB_MIS}" \
    "${CODELAB_HDD_USER_ROOT}/${user_name}/${CODELAB_MIS}"; do
    if [[ -d "$candidate" && -w "$candidate" ]]; then
      CODELAB_PERSIST_DIR="$candidate"
      return
    fi
  done

  while read -r target source; do
    mount_source="${source#*[}"
    mount_source="${mount_source%]}"
    [[ "${mount_source##*/}" == "$CODELAB_MIS" ]] || continue
    if [[ -d "$target" && -w "$target" ]]; then
      CODELAB_PERSIST_DIR="$target"
      return
    fi
  done < <(findmnt -rn -t fuse.bgfuse -o TARGET,SOURCE 2>/dev/null)

  die "Unable to find a writable persistence directory."
  return 1
}

init_persist_env() {
  if [[ -n "${CODELAB_PERSIST_DIR:-}" ]]; then
    detect_persist_dir || return 1
    if [[ -n "${CODELAB_MIS:-}" ]]; then
      detect_mis || return 1
    else
      detect_mis >/dev/null 2>&1 || true
    fi
  else
    detect_mis || return 1
    detect_persist_dir || return 1
  fi

  export CODELAB_PERSIST_DIR
  [[ -z "${CODELAB_MIS:-}" ]] || export CODELAB_MIS
}

project_name() {
  local workspace="$1" name
  name="${CODELAB_PROJECT_NAME:-$(basename "${workspace%/}")}"
  valid_name "$name" || {
    die "Project name has an invalid format."
    return 1
  }
  printf '%s\n' "$name"
}

backup_dir() {
  local workspace="$1" name
  name="$(project_name "$workspace")"
  printf '%s/.workspace-recovery/%s/worktree\n' "$CODELAB_PERSIST_DIR" "$name"
}

run_rsync() {
  if [[ "${CODELAB_PERSIST_DEBUG:-0}" == 1 ]]; then
    rsync "$@"
  else
    rsync "$@" >/dev/null 2>&1
  fi
}

save_workspace() {
  local workspace="$1" destination
  [[ -d "$workspace" ]] || {
    die "Workspace is not a directory."
    return 1
  }
  command -v rsync >/dev/null 2>&1 || {
    die "rsync is required."
    return 1
  }

  destination="$(backup_dir "$workspace")" || return 1
  mkdir -p "$destination" || {
    die "Unable to create the private backup directory."
    return 1
  }
  chmod 700 "${CODELAB_PERSIST_DIR}/.workspace-recovery" \
    "$(dirname "$destination")" "$destination" || {
    die "Unable to secure the private backup directory."
    return 1
  }

  run_rsync -a \
    --exclude='.env' \
    --exclude='.env.*' \
    --exclude='.ssh/' \
    --exclude='.codex/' \
    --exclude='*.key' \
    --exclude='*.pem' \
    "$workspace/" "$destination/" || {
    die "Backup failed."
    return 1
  }

  printf 'Backup completed.\n'
}

restore_workspace() {
  local workspace="$1" force="$2" source
  command -v rsync >/dev/null 2>&1 || {
    die "rsync is required."
    return 1
  }
  source="$(backup_dir "$workspace")" || return 1
  [[ -d "$source" ]] || {
    die "No backup was found for this project."
    return 1
  }

  if [[ "$force" != 1 && -d "$workspace" &&
    -n "$(find "$workspace" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
    die "Destination is not empty; use restore --force to merge and overwrite files."
    return 1
  fi

  mkdir -p "$workspace" || {
    die "Unable to create the destination directory."
    return 1
  }
  run_rsync -a "$source/" "$workspace/" || {
    die "Restore failed."
    return 1
  }
  printf 'Restore completed.\n'
}

usage() {
  cat <<'EOF'
Usage:
  source tools/env/codelab_persist.sh
  tools/env/codelab_persist.sh [options] check
  tools/env/codelab_persist.sh [options] save [workspace]
  tools/env/codelab_persist.sh [options] restore [--force] [workspace]

Options:
  --persist-dir PATH  Use PATH instead of automatic Codelab detection.
  --mis MIS           Override automatic MIS detection.
  --project NAME      Use the same backup name across different workspace paths.
  -h, --help          Show this help.

Environment overrides:
  CODELAB_PERSIST_DIR, CODELAB_MIS, CODELAB_PROJECT_NAME
  CODELAB_SSD_USER_ROOT, CODELAB_HDD_USER_ROOT
  CODELAB_PRIVATE_MOUNT_PREFIX

Set CODELAB_PERSIST_DEBUG=1 only when path-bearing rsync diagnostics are safe.
EOF
}

main() {
  local command_name workspace force=0

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --persist-dir)
        [[ $# -ge 2 ]] || {
          die "--persist-dir requires a path."
          return 2
        }
        CODELAB_PERSIST_DIR="$2"
        shift 2
        ;;
      --persist-dir=*)
        CODELAB_PERSIST_DIR="${1#*=}"
        shift
        ;;
      --mis)
        [[ $# -ge 2 ]] || {
          die "--mis requires a value."
          return 2
        }
        CODELAB_MIS="$2"
        shift 2
        ;;
      --mis=*)
        CODELAB_MIS="${1#*=}"
        shift
        ;;
      --project)
        [[ $# -ge 2 ]] || {
          die "--project requires a value."
          return 2
        }
        CODELAB_PROJECT_NAME="$2"
        shift 2
        ;;
      --project=*)
        CODELAB_PROJECT_NAME="${1#*=}"
        shift
        ;;
      --)
        shift
        break
        ;;
      -h|--help)
        usage
        return
        ;;
      *)
        break
        ;;
    esac
  done

  command_name="${1:-}"

  case "$command_name" in
    check)
      init_persist_env || return 1
      printf 'Writable persistence storage detected.\n'
      ;;
    save)
      init_persist_env || return 1
      workspace="${2:-$PWD}"
      save_workspace "$workspace"
      ;;
    restore)
      init_persist_env || return 1
      shift
      if [[ "${1:-}" == --force ]]; then
        force=1
        shift
      fi
      workspace="${1:-$PWD}"
      restore_workspace "$workspace" "$force"
      ;;
    help|'')
      usage
      ;;
    *)
      usage >&2
      return 2
      ;;
  esac
}

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  init_persist_env
else
  set -euo pipefail
  umask 077
  main "$@"
fi
