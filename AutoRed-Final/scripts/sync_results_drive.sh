#!/usr/bin/env bash
#
# sync_results_drive.sh — archive the current results/ tree to a dated zip and
# push it to Google Drive, then delete the local zip.
#
# WHAT IT DOES
#   1. Zips the local AutoRed-Final/results/ tree (the new results-layout output:
#      results/benchmark/<model>/<chars>/{logs,runs}/...) into a dated file:
#        results_<YYYYMMDD_HHMMSS>.zip
#   2. Uploads that zip to gdrive:<REMOTE_RESULTS_DIR>/ via rclone.
#   3. Deletes the local zip (Drive is now the source of truth for that archive).
#
# By default only results/benchmark/ (the canonical output) is zipped, NOT
# results_bak/ (the 1.4G legacy archive). Use --include-bak to also archive
# results_bak/ (large; slow over Drive).
#
# Run this AFTER a benchmark finishes (e.g. call it from the tail of
# hpc/autored_benchmark_4gpu_vllm.sh, or manually). One-directional: local→Drive.
#
# USAGE
#   ./sync_results_drive.sh                  # archive results/ → Drive, rm local zip
#   ./sync_results_drive.sh --include-bak    # also include results_bak/
#   ./sync_results_drive.sh --dry-run        # zip + show what would be pushed, don't upload
#   ./sync_results_drive.sh --help
#
# REQUIRES: rclone configured with a remote named 'gdrive' (run `rclone config`).

set -euo pipefail

# ---------------------------------------------------------------------------
# Config (override via env vars if needed)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTORED_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"           # AutoRed-Final/
RESULTS_DIR="${AUTORED_DIR}/results"                  # new-layout output tree
REMOTE="gdrive"
REMOTE_RESULTS_DIR="AutoRed-Combination/results"      # gdrive:<this>/results_*.zip
# Pin the rclone root to a *shared* Google Drive folder by ID so both you and a
# coworker (each using their own 'gdrive' account) land in the SAME shared
# folder regardless of whose Drive root it sits under. The folder ID is
# constant; the path (AutoRed-Combination/results) is resolved relative to it.
# Override per-invocation with --root-folder-id, or change the default here.
DRIVE_ROOT_FOLDER_ID="14TP-ANowJkqYMLJsPFcdQz_ZVB4wPytz"
STAMP="$(date +%Y%m%d_%H%M%S)"
ZIP_NAME="results_${STAMP}.zip"

# Flags appended to EVERY rclone call (kept as an array to survive --dry-run
# previews and to stay quoting-safe). Empty if no root-folder pin is set.
RCLONE_ROOT_FLAGS=()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
print_help() {
  cat << 'HELP'
sync_results_drive.sh — archive results/ to a dated zip and push to Google Drive.

USAGE:
  ./sync_results_drive.sh [OPTIONS]

OPTIONS:
  --include-bak     Also archive results_bak/ (the legacy 1.4G backup). Off by
                    default — only results/ (the new-layout benchmark output) is
                    zipped, keeping each archive small.
  --dry-run         Build the zip and show what would be uploaded, but skip the
                    actual rclone upload and the local-zip cleanup.
  --remote NAME     rclone remote to use (default: gdrive).
  --remote-dir P     Remote directory under the pinned folder (default: AutoRed-Combination/results).
  --root-folder-id  Google Drive folder ID to pin as rclone's root. Defaults to
                    a SHARED folder so you and a coworker (each on your own
                    'gdrive' account) write to the same place. Pass a different
                    ID to target another folder; pass '' (empty) to use the
                    remote's own Drive root instead.
  --help, -h        Show this help.

ENV:
  Results dir:  AutoRed-Final/results/  (new layout: benchmark/<model>/<chars>/...)
  Remote path:  gdrive:AutoRed-Combination/results/<results_TIMESTAMP>.zip
                (resolved relative to the pinned shared folder — see --root-folder-id)

NOTES:
  - The local zip is deleted after a confirmed upload (Drive = source of truth).
  - Use --dry-run to preview the archive size and remote path without uploading.
  - Requires rclone with a 'gdrive' remote (run `rclone config` to set up).
  - Both you and your coworker must have access to the pinned shared folder.
HELP
}

INCLUDE_BAK=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --include-bak)   INCLUDE_BAK=1; shift ;;
    --dry-run)       DRY_RUN=1; shift ;;
    --remote)        REMOTE="$2"; shift 2 ;;
    --remote-dir)    REMOTE_RESULTS_DIR="$2"; shift 2 ;;
    --root-folder-id) DRIVE_ROOT_FOLDER_ID="$2"; shift 2 ;;
    --help|-h)       print_help; exit 0 ;;
    *) echo "error: unknown option: $1" >&2; echo "try --help" >&2; exit 2 ;;
  esac
done

# Build the root-pinning flag array after arg parsing so an explicit
# --root-folder-id '' (empty) disables pinning (uses the remote's own root).
if [[ -n "$DRIVE_ROOT_FOLDER_ID" ]]; then
  RCLONE_ROOT_FLAGS=(--drive-root-folder-id "$DRIVE_ROOT_FOLDER_ID")
else
  RCLONE_ROOT_FLAGS=()
fi

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
if ! command -v rclone >/dev/null 2>&1; then
  echo "error: rclone not found. Install it and run 'rclone config' to add a 'gdrive' remote." >&2
  exit 1
fi
if ! command -v zip >/dev/null 2>&1; then
  echo "error: zip not found." >&2; exit 1
fi
if [[ ! -d "$RESULTS_DIR" ]]; then
  echo "error: results dir not found: $RESULTS_DIR" >&2
  exit 1
fi
if [[ ! -d "$RESULTS_DIR/benchmark" ]]; then
  echo "warn: $RESULTS_DIR/benchmark/ not found — nothing to archive?" >&2
  echo "       (the new layout writes to results/benchmark/<model>/<chars>/)" >&2
fi

ZIP_PATH="${AUTORED_DIR}/${ZIP_NAME}"

# ---------------------------------------------------------------------------
# Build the zip
# ---------------------------------------------------------------------------
echo "[1/3] Archiving results/ → ${ZIP_NAME}"
# Zip relative to AUTORED_DIR so the archive root is 'results/' (+ optional
# 'results_bak/'). -q for quiet, -r for recursive, -y to store symlinks as-is.
ZIP_ARGS=(-qr "$ZIP_PATH")
ZIP_TARGETS=("results")
if [[ "$INCLUDE_BAK" -eq 1 ]]; then
  if [[ -d "${AUTORED_DIR}/results_bak" ]]; then
    ZIP_TARGETS+=("results_bak")
    echo "      (including results_bak/ — archive may be large)"
  else
    echo "      --include-bak given but results_bak/ missing; skipping it"
  fi
fi

( cd "$AUTORED_DIR" && zip "${ZIP_ARGS[@]}" "${ZIP_TARGETS[@]}" )

if [[ ! -f "$ZIP_PATH" ]]; then
  echo "error: zip creation failed — $ZIP_PATH not found" >&2
  exit 1
fi
ZIP_SIZE=$(du -h "$ZIP_PATH" | cut -f1)
echo "      zip size: ${ZIP_SIZE}"
echo "      file count in results/: $(find "$RESULTS_DIR" -type f | wc -l)"

# ---------------------------------------------------------------------------
# Upload (unless dry-run)
# ---------------------------------------------------------------------------
REMOTE_PATH="${REMOTE}:${REMOTE_RESULTS_DIR}/${ZIP_NAME}"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[2/3] (dry-run) would run: rclone copy \"${ZIP_PATH}\" \"${REMOTE_PATH}\" --progress ${RCLONE_ROOT_FLAGS[*]}"
  echo "      (zip kept locally: ${ZIP_PATH})"
  echo "      (no upload, no local cleanup)"
  echo "[3/3] done (dry-run)"
  exit 0
fi

echo "[2/3] Uploading to ${REMOTE_PATH}"
if [[ "${#RCLONE_ROOT_FLAGS[@]}" -gt 0 ]]; then
  echo "      (pinned to shared folder id: ${DRIVE_ROOT_FOLDER_ID})"
fi
if ! rclone copy "${ZIP_PATH}" "${REMOTE_PATH}" --progress "${RCLONE_ROOT_FLAGS[@]}"; then
  echo "error: rclone upload failed. Local zip preserved: ${ZIP_PATH}" >&2
  echo "       re-run, or upload manually." >&2
  exit 1
fi

# Verify the upload by checking the remote object exists.
if ! rclone lsf "${REMOTE}:${REMOTE_RESULTS_DIR}/${ZIP_NAME}" "${RCLONE_ROOT_FLAGS[@]}" >/dev/null 2>&1; then
  echo "error: upload verification failed — ${ZIP_NAME} not found on remote." >&2
  echo "       local zip preserved: ${ZIP_PATH}" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Clean up local zip
# ---------------------------------------------------------------------------
echo "[3/3] Upload verified — deleting local zip"
rm -f "$ZIP_PATH"
echo "done. Archived ${ZIP_NAME} to ${REMOTE_PATH}"
echo
if [[ "${#RCLONE_ROOT_FLAGS[@]}" -gt 0 ]]; then
  echo "List archives:  rclone lsf ${REMOTE}:${REMOTE_RESULTS_DIR}/ --drive-root-folder-id ${DRIVE_ROOT_FOLDER_ID}"
else
  echo "List archives:  rclone lsf ${REMOTE}:${REMOTE_RESULTS_DIR}/"
fi
echo "Pull latest:     ./sync_results_local.sh"
echo "Pull a specific: ./sync_results_local.sh results_${STAMP}.zip"
