#!/usr/bin/env bash
#
# sync_results_local.sh — pull a results archive from Google Drive, unzip it,
# and merge it into the local results/ tree (with deduplication), or replace
# the local tree.
#
# DEFAULT BEHAVIOR
#   Pull the LATEST results_*.zip on Drive, unzip to a temp dir, and MERGE its
#   results/ contents into AutoRed-Final/results/. Merge is deduplicated:
#   when a file exists at the same relative path locally, the NEWER file wins
#   (by mtime); when no collision, the pulled file is added. This makes repeat
#   pulls idempotent — you can pull the same archive twice safely.
#
# USAGE
#   ./sync_results_local.sh                         # latest zip, MERGE into results/
#   ./sync_results_local.sh --replace                # latest zip, REPLACE results/ (destructive)
#   ./sync_results_local.sh results_20260807_024123.zip   # pull a specific dated zip, MERGE
#   ./sync_results_local.sh results_20260807_024123.zip --replace
#   ./sync_results_local.sh --list                   # list available archives on Drive, don't pull
#   ./sync_results_local.sh --dry-run                # show what would happen, don't write
#   ./sync_results_local.sh --help
#
# The <zip> positional arg (if given) must be a bare filename (e.g.
# results_20260807_024123.zip), matched against the remote dir. A path with
# slashes is rejected — pass just the filename.
#
# REQUIRES: rclone configured with a remote named 'gdrive'.
# REQUIRES: rsync (used for the dedup-aware merge: --update = newer wins).

set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTORED_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"           # AutoRed-Final/
RESULTS_DIR="${AUTORED_DIR}/results"                  # new-layout output tree
REMOTE="gdrive"
REMOTE_RESULTS_DIR="AutoRed-Combination/results"      # gdrive:<this>/results_*.zip
# Pin the rclone root to a *shared* Google Drive folder by ID so both you and a
# coworker (each using their own 'gdrive' account) pull from the SAME shared
# folder regardless of whose Drive root it sits under. The folder ID is
# constant; the path (AutoRed-Combination/results) is resolved relative to it.
# Override per-invocation with --root-folder-id, or change the default here.
DRIVE_ROOT_FOLDER_ID="14TP-ANowJkqYMLJsPFcdQz_ZVB4wPytz"
TMP_BASE="${TMPDIR:-/tmp}"

# Flags appended to EVERY rclone call (kept as an array to survive --dry-run
# previews and to stay quoting-safe). Empty if no root-folder pin is set.
RCLONE_ROOT_FLAGS=()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
print_help() {
  cat << 'HELP'
sync_results_local.sh — pull a results archive from Drive, unzip, merge/replace.

USAGE:
  ./sync_results_local.sh [OPTIONS] [ZIP]

ARGUMENTS:
  ZIP             A bare results_<TIMESTAMP>.zip filename on the remote dir to
                  pull (e.g. results_20260807_024123.zip). If omitted, the
                  latest archive on Drive is pulled automatically.

OPTIONS:
  --replace       REPLACE the local results/ tree with the archive contents
                  (destructive: removes existing results/ first). Default is
                  MERGE (dedup: newer file wins on collision).
  --merge         Explicitly request merge mode (the default). Newer file wins
                  on path collision via rsync --update.
  --list          List available archives on the remote dir and exit (no pull).
  --dry-run       Show what would happen (which zip, merge/replace, paths)
                  without writing to results/.
  --remote NAME   rclone remote to use (default: gdrive).
  --remote-dir P  Remote directory under the pinned folder (default: AutoRed-Combination/results).
  --root-folder-id  Google Drive folder ID to pin as rclone's root. Defaults to
                    a SHARED folder so you and a coworker (each on your own
                    'gdrive' account) pull from the same place. Pass a different
                    ID to target another folder; pass '' (empty) to use the
                    remote's own Drive root instead.
  --help, -h      Show this help.

ENV:
  Results dir:    AutoRed-Final/results/  (new layout: benchmark/<model>/<chars>/...)
  Remote path:    gdrive:AutoRed-Combination/results/
                  (resolved relative to the pinned shared folder — see --root-folder-id)
  Merge tool:     rsync --update (newer mtime wins on collision)

NOTES:
  - Merge (default) is idempotent: pulling the same archive twice is safe.
  - Replace removes the existing results/ tree first, then extracts. Back it
    up yourself first if you need to (the tree is gitignored, so git won't
    recover it).
  - Only results/ is affected by merge/replace. The archive's results_bak/
    (if present in the zip) is extracted to results_bak/ under the same
    merge/replace policy.
  - Both you and your coworker must have access to the pinned shared folder.
HELP
}

list_remote_archives() {
  # Print remote results_*.zip FILE names, one per line, sorted (oldest→newest).
  # --files-only excludes directories, so a stray folder named like a zip
  # (the earlier rclone-copy-into-path bug) can't masquerade as an archive.
  rclone lsf "${REMOTE}:${REMOTE_RESULTS_DIR}/" --files-only "${RCLONE_ROOT_FLAGS[@]}" 2>/dev/null \
    | grep -E '^results_[0-9_]+\.zip$' || true
}

latest_remote_zip() {
  # Filenames are results_<YYYYMMDD_HHMMSS>.zip — lexicographic sort == time sort.
  list_remote_archives | tail -1
}

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
MODE="merge"        # or "replace"
LIST_ONLY=0
DRY_RUN=0
ZIP_ARG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --replace)    MODE="replace"; shift ;;
    --merge)      MODE="merge"; shift ;;
    --list)       LIST_ONLY=1; shift ;;
    --dry-run)    DRY_RUN=1; shift ;;
    --remote)        REMOTE="$2"; shift 2 ;;
    --remote-dir)    REMOTE_RESULTS_DIR="$2"; shift 2 ;;
    --root-folder-id) DRIVE_ROOT_FOLDER_ID="$2"; shift 2 ;;
    --help|-h)       print_help; exit 0 ;;
    --*)             echo "error: unknown option: $1" >&2; echo "try --help" >&2; exit 2 ;;
    *)               ZIP_ARG="$1"; shift ;;
  esac
done

# Build the root-pinning flag array after arg parsing so an explicit
# --root-folder-id '' (empty) disables pinning (uses the remote's own root).
if [[ -n "$DRIVE_ROOT_FOLDER_ID" ]]; then
  RCLONE_ROOT_FLAGS=(--drive-root-folder-id "$DRIVE_ROOT_FOLDER_ID")
else
  RCLONE_ROOT_FLAGS=()
fi

if ! command -v rclone >/dev/null 2>&1; then
  echo "error: rclone not found. Install it and run 'rclone config'." >&2; exit 1
fi
if ! command -v rsync >/dev/null 2>&1; then
  echo "error: rsync not found (used for dedup-aware merge)." >&2; exit 1
fi
if ! command -v unzip >/dev/null 2>&1; then
  echo "error: unzip not found." >&2; exit 1
fi

# Scope guard: rclone defaults to the restrictive 'drive.file' scope when scope
# is unset, which lets it see ONLY files created by this own rclone instance —
# it CANNOT see the contents of a folder shared by another account (e.g. the
# pinned shared results folder pushed from a coworker's account). That is the
# classic "no results*.zip on the drive" on a second machine. We need the full
# 'drive' scope to list the shared folder. Warn clearly before the pull fails.
remote_scope="$(rclone config show "$REMOTE" 2>/dev/null | awk -F'=' '/^[[:space:]]*scope[[:space:]]*=/{gsub(/[[:space:]]/,"",$2);print $2}')"
if [[ -z "$remote_scope" ]]; then
  remote_scope="drive.file"   # rclone's default when scope is absent
fi
if [[ "$remote_scope" != "drive" ]]; then
  echo "WARNING: rclone remote '${REMOTE}' has scope '${remote_scope}' (not 'drive')." >&2
  echo "         scope='${remote_scope}' only sees files created by this rclone instance —" >&2
  echo "         it CANNOT list a folder shared by another account (e.g. the pinned" >&2
  echo "         shared results folder). If the pull below fails with 'no results*.zip'," >&2
  echo "         fix the remote on THIS machine with:" >&2
  echo "             rclone config        # edit '${REMOTE}' → advanced → scope = drive" >&2
  echo "         then re-authorize and re-run this script. Also confirm the shared folder" >&2
  echo "         ${DRIVE_ROOT_FOLDER_ID} is shared with this Google account." >&2
  echo >&2
fi

# ---------------------------------------------------------------------------
# --list: just show archives and exit
# ---------------------------------------------------------------------------
if [[ "$LIST_ONLY" -eq 1 ]]; then
  echo "Archives in ${REMOTE}:${REMOTE_RESULTS_DIR}/"
  if [[ "${#RCLONE_ROOT_FLAGS[@]}" -gt 0 ]]; then
    echo "(pinned to shared folder id: ${DRIVE_ROOT_FOLDER_ID})"
  fi
  archives="$(list_remote_archives)"
  if [[ -z "$archives" ]]; then
    echo "  (none found)"
  else
    echo "$archives" | sed 's/^/  /'
  fi
  exit 0
fi

# ---------------------------------------------------------------------------
# Decide which zip to pull
# ---------------------------------------------------------------------------
if [[ -n "$ZIP_ARG" ]]; then
  if [[ "$ZIP_ARG" == */* ]]; then
    echo "error: ZIP must be a bare filename, not a path: $ZIP_ARG" >&2
    echo "       pass just 'results_<TIMESTAMP>.zip' (matched against the remote dir)." >&2
    exit 2
  fi
  ZIP_NAME="$ZIP_ARG"
else
  ZIP_NAME="$(latest_remote_zip)"
  if [[ -z "$ZIP_NAME" ]]; then
    echo "error: no results_*.zip archives found in ${REMOTE}:${REMOTE_RESULTS_DIR}/" >&2
    if [[ "${#RCLONE_ROOT_FLAGS[@]}" -gt 0 ]]; then
      echo "       (pinned to shared folder id: ${DRIVE_ROOT_FOLDER_ID})" >&2
    fi
    echo "       run ./sync_results_drive.sh first to push one." >&2
    exit 1
  fi
  echo "Latest archive detected: $ZIP_NAME"
fi

REMOTE_PATH="${REMOTE}:${REMOTE_RESULTS_DIR}/${ZIP_NAME}"
echo "Pulling:  $REMOTE_PATH"
if [[ "${#RCLONE_ROOT_FLAGS[@]}" -gt 0 ]]; then
  echo "         (pinned to shared folder id: ${DRIVE_ROOT_FOLDER_ID})"
fi

# ---------------------------------------------------------------------------
# Dry-run preview
# ---------------------------------------------------------------------------
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[dry-run] mode:     $MODE"
  echo "[dry-run] target:   $RESULTS_DIR"
  echo "[dry-run] would:    download $ZIP_NAME → temp, unzip, ${MODE} into results/"
  echo "[dry-run] rclone:   rclone copy \"$REMOTE_PATH\" <tmp>/ --progress ${RCLONE_ROOT_FLAGS[*]}"
  echo "[dry-run] no changes made."
  exit 0
fi

# ---------------------------------------------------------------------------
# Download + unzip to a temp dir
# ---------------------------------------------------------------------------
WORKDIR="$(mktemp -d "${TMP_BASE}/autored_results.XXXXXX")"
trap 'rm -rf "$WORKDIR"' EXIT
ZIP_LOCAL="${WORKDIR}/${ZIP_NAME}"

echo "[1/4] Downloading ${ZIP_NAME} → ${ZIP_LOCAL}"
if ! rclone copy "$REMOTE_PATH" "$WORKDIR/" --progress "${RCLONE_ROOT_FLAGS[@]}"; then
  echo "error: rclone download failed for $REMOTE_PATH" >&2
  exit 1
fi
if [[ ! -f "$ZIP_LOCAL" ]]; then
  echo "error: downloaded zip not found at $ZIP_LOCAL" >&2
  exit 1
fi

echo "[2/4] Unzipping → ${WORKDIR}/"
if ! unzip -q "$ZIP_LOCAL" -d "$WORKDIR"; then
  echo "error: unzip failed for $ZIP_LOCAL" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Merge or replace into results/
# ---------------------------------------------------------------------------
# bash 3.2 (macOS default) does not support ${MODE^} case-modification; use
# an explicit ternary so the script runs on both macOS bash 3.2 and Linux bash 4+.
if [[ "$MODE" == "merge" ]]; then
  echo "[3/4] Merge into ${RESULTS_DIR}"
else
  echo "[3/4] Replace into ${RESULTS_DIR}"
fi

# Helper: apply merge or replace for one top-level tree (e.g. "results", "results_bak")
apply_tree() {
  local tree="$1"   # relative dir name inside the zip, e.g. "results"
  local src="${WORKDIR}/${tree}"
  local dst="${AUTORED_DIR}/${tree}"

  if [[ ! -d "$src" ]]; then
    echo "      (no ${tree}/ in archive — skipping)"
    return 0
  fi

  if [[ "$MODE" == "replace" ]]; then
    if [[ -d "$dst" ]]; then
      echo "      removing existing ${tree}/ (replace mode)"
      rm -rf "$dst"
    fi
    mkdir -p "$dst"
    # Plain copy — dst was just emptied, so no collision possible.
    cp -a "${src}/." "$dst/"
  else
    # MERGE with dedup: rsync --update copies files that are newer on the
    # source OR missing on the destination. Existing newer local files are
    # kept; collisions resolve to newer mtime. -a preserves perms/structure,
    # --inplace avoids temp-file churn for large merged runs.
    mkdir -p "$dst"
    rsync -a --update --inplace --human-readable "${src}/" "$dst/"
  fi
}

apply_tree "results"
# results_bak/ is huge and rarely in a benchmark archive, but handle it if present.
apply_tree "results_bak"

if [[ "$MODE" == "merge" ]]; then
  echo "[4/4] Done. Merge complete."
else
  echo "[4/4] Done. Replace complete."
fi
echo "      results/ now contains: $(find "$RESULTS_DIR" -type f 2>/dev/null | wc -l) files"
echo
echo "Local zip was in a temp dir (auto-removed). Drive still holds: ${REMOTE_PATH}"
