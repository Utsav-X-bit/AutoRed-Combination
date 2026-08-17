#!/usr/bin/env bash
#
# sync_results_drive.sh — stage results/ onto fast RAID storage, zip it there,
# push the zip to Google Drive, then delete the staged tree + zip from the RAID.
#
# WHY RAID STAGING
#   Zipping directly on the network filesystem (NFS) is too slow. So the tree
#   is first rsync'd to a local RAID volume, zipped there (fast), pushed to
#   Drive, and the RAID copies are removed once the upload is verified.
#
# WHAT IT DOES
#   1. rsync's AutoRed-Final/results/ → $RAID_STAGING/results/ (fast local disk).
#   2. (with --merge-bak) also rsync's AutoRed-Final/results_bak/ →
#      $RAID_STAGING/results_bak/ (the 1.4G legacy archive; off by default).
#   3. Zips each staged tree ON THE RAID into dated files:
#        results_<YYYYMMDD_HHMMSS>.zip          (always)
#        results_bak_<YYYYMMDD_HHMMSS>.zip      (only with --merge-bak)
#   4. Uploads each zip to gdrive:<REMOTE_RESULTS_DIR>/ via rclone, verifying
#      the file actually landed on the remote.
#   5. After each verified upload: deletes that zip AND its staged tree from
#      the RAID (Drive is now the source of truth for that archive).
#
# Run this AFTER a benchmark finishes (e.g. call it from the tail of
# hpc/autored_benchmark_4gpu_vllm.sh, or manually). One-directional: local→Drive.
#
# USAGE
#   ./sync_results_drive.sh                  # results/ → RAID → zip → Drive → clean RAID
#   ./sync_results_drive.sh --merge-bak      # also copy + zip + push results_bak/
#   ./sync_results_drive.sh --dry-run        # stage + zip on RAID, show what would be
#                                            # pushed, don't upload, keep RAID copies
#   ./sync_results_drive.sh --help
#
# REQUIRES: rclone (remote 'gdrive'), rsync, zip.

set -euo pipefail

# ---------------------------------------------------------------------------
# Config (override via env vars if needed)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTORED_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"           # AutoRed-Final/
RESULTS_DIR="${AUTORED_DIR}/results"                  # new-layout output tree
BAK_DIR="${AUTORED_DIR}/results_bak"                  # legacy 1.4G backup tree
# Fast local staging volume (zip happens HERE, not on the network fs).
RAID_STAGING_DIR="${RAID_STAGING_DIR:-/raid/user-45839/AutoRed-Combination}"
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
BAK_ZIP_NAME="results_bak_${STAMP}.zip"

# Flags appended to EVERY rclone call (kept as an array to survive --dry-run
# previews and to stay quoting-safe). Empty if no root-folder pin is set.
RCLONE_ROOT_FLAGS=()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
print_help() {
  cat << 'HELP'
sync_results_drive.sh — stage results/ on fast RAID, zip there, push to Drive,
then clean the RAID. (Zipping directly on network storage is too slow.)

USAGE:
  ./sync_results_drive.sh [OPTIONS]

OPTIONS:
  --merge-bak       Also rsync + zip + push results_bak/ (the legacy 1.4G
                    backup) as results_bak_<timestamp>.zip. Off by default —
                    only results/ (the new-layout benchmark output) is archived,
                    keeping the main archive small.
  --dry-run         Stage + zip on the RAID and show what would be uploaded,
                    but skip the rclone upload and the RAID cleanup (RAID
                    copies are kept).
  --remote NAME     rclone remote to use (default: gdrive).
  --remote-dir P    Remote directory under the pinned folder (default: AutoRed-Combination/results).
  --root-folder-id  Google Drive folder ID to pin as rclone's root. Defaults to
                    a SHARED folder so you and a coworker (each on your own
                    'gdrive' account) write to the same place. Pass a different
                    ID to target another folder; pass '' (empty) to use the
                    remote's own Drive root instead.
  --help, -h        Show this help.

ENV:
  Results dir:   AutoRed-Final/results/       (new layout: benchmark/<model>/<chars>/...)
  Bak dir:       AutoRed-Final/results_bak/   (legacy, only with --merge-bak)
  RAID staging:  /raid/user-45839/AutoRed-Combination
                 (override with RAID_STAGING_DIR=/path)
  Remote path:   gdrive:AutoRed-Combination/results/<results_TIMESTAMP>.zip
                 (resolved relative to the pinned shared folder — see --root-folder-id)

NOTES:
  - Staged trees + zips are deleted from the RAID only after the upload is
    verified. On any failure the RAID copies are preserved for a re-push.
  - rsync uses --delete so the RAID staging is an exact mirror of the source
    (stale files from a previous failed run do not leak into the archive).
  - Requires rclone with a 'gdrive' remote (run `rclone config` to set up),
    plus rsync and zip.
  - Both you and your coworker must have access to the pinned shared folder.
HELP
}

# Upload one zip to Drive, verify it, then clean it + its staged tree(s) from
# the RAID. Usage: push_zip <zip-path> <staged-tree> [staged-tree ...]
# On failure: prints an error, preserves the RAID copies, and exits 1.
push_zip() {
  local zip_path="$1"; shift
  local zip_name cleanup
  zip_name="$(basename "$zip_path")"
  local remote_dir="${REMOTE}:${REMOTE_RESULTS_DIR}/"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "    (dry-run) would run: rclone copy \"${zip_path}\" \"${remote_dir}\" --progress ${RCLONE_ROOT_FLAGS[*]}"
    for cleanup in "$@"; do
      echo "    (dry-run) keeping on RAID: ${zip_path} and ${cleanup}"
    done
    return 0
  fi

  echo "    Uploading ${zip_name} → ${remote_dir}"
  if ! rclone copy "${zip_path}" "${remote_dir}" --progress "${RCLONE_ROOT_FLAGS[@]}"; then
    echo "error: rclone upload failed for ${zip_name}. RAID copies preserved:" >&2
    echo "       zip: ${zip_path}" >&2
    for cleanup in "$@"; do echo "       staged: ${cleanup}" >&2; done
    echo "       re-run this script (rsync is incremental), or push manually." >&2
    exit 1
  fi

  # Verify the upload: confirm ${zip_name} exists as a FILE in the remote dir.
  # --files-only prevents a folder-with-the-same-name from passing as
  # "uploaded" (the earlier bug: rclone made results_<ts>.zip/ a directory).
  if ! rclone lsf "${remote_dir}" --files-only "${RCLONE_ROOT_FLAGS[@]}" 2>/dev/null | grep -qx "${zip_name}"; then
    echo "error: upload verification failed — ${zip_name} not found as a file on remote." >&2
    echo "       (if a directory named ${zip_name} exists, delete it: rclone purge \"${REMOTE}:${REMOTE_RESULTS_DIR}/${zip_name}/\" ${RCLONE_ROOT_FLAGS[*]})" >&2
    echo "       RAID copies preserved: ${zip_path}" >&2
    exit 1
  fi

  echo "    Upload verified — removing from RAID: ${zip_name} + staged tree(s)"
  rm -f "${zip_path}"
  for cleanup in "$@"; do
    rm -rf "${cleanup}"
  done
}

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
MERGE_BAK=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --merge-bak)     MERGE_BAK=1; shift ;;
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
for tool in rclone rsync zip; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "error: $tool not found (required by this script)." >&2
    exit 1
  fi
done
if [[ ! -d "$RESULTS_DIR" ]]; then
  echo "error: results dir not found: $RESULTS_DIR" >&2
  exit 1
fi
if [[ ! -d "$RESULTS_DIR/benchmark" ]]; then
  echo "warn: $RESULTS_DIR/benchmark/ not found — nothing to archive?" >&2
  echo "      (the new layout writes to results/benchmark/<model>/<chars>/)" >&2
fi
if ! mkdir -p "$RAID_STAGING_DIR"; then
  echo "error: cannot create RAID staging dir: $RAID_STAGING_DIR" >&2
  exit 1
fi
if [[ ! -w "$RAID_STAGING_DIR" ]]; then
  echo "error: RAID staging dir not writable: $RAID_STAGING_DIR" >&2
  exit 1
fi

RAID_RESULTS="${RAID_STAGING_DIR}/results"
RAID_BAK="${RAID_STAGING_DIR}/results_bak"
ZIP_PATH="${RAID_STAGING_DIR}/${ZIP_NAME}"
BAK_ZIP_PATH="${RAID_STAGING_DIR}/${BAK_ZIP_NAME}"

# ---------------------------------------------------------------------------
# [1/3] Stage on RAID (fast local disk) via rsync
# ---------------------------------------------------------------------------
echo "[1/3] Staging on RAID: ${RAID_STAGING_DIR}/"
echo "      rsync results/ → ${RAID_RESULTS}/"
# --delete: make the staging dir an exact mirror so stale files from a
# previous interrupted run cannot leak into the archive.
rsync -a --delete --info=progress2 "$RESULTS_DIR" "$RAID_STAGING_DIR/"

if [[ "$MERGE_BAK" -eq 1 ]]; then
  if [[ -d "$BAK_DIR" ]]; then
    echo "      rsync results_bak/ → ${RAID_BAK}/"
    rsync -a --delete --info=progress2 "$BAK_DIR" "$RAID_STAGING_DIR/"
  else
    echo "      --merge-bak given but $BAK_DIR missing; skipping it"
    MERGE_BAK=0
  fi
fi

# ---------------------------------------------------------------------------
# [2/3] Zip ON THE RAID (not on network storage)
# ---------------------------------------------------------------------------
echo "[2/3] Zipping on RAID (fast local disk)"
# Zip relative to the staging dir so the archive root is 'results/'
# (+ optional 'results_bak/'). -q quiet, -r recursive, -y store symlinks as-is.
( cd "$RAID_STAGING_DIR" && zip -qry "${ZIP_NAME}" results )
if [[ ! -f "$ZIP_PATH" ]]; then
  echo "error: zip creation failed — $ZIP_PATH not found" >&2
  exit 1
fi
echo "      ${ZIP_NAME}: $(du -h "$ZIP_PATH" | cut -f1)"
echo "      file count in results/: $(find "$RESULTS_DIR" -type f | wc -l)"

if [[ "$MERGE_BAK" -eq 1 ]]; then
  ( cd "$RAID_STAGING_DIR" && zip -qry "${BAK_ZIP_NAME}" results_bak )
  if [[ ! -f "$BAK_ZIP_PATH" ]]; then
    echo "error: bak zip creation failed — $BAK_ZIP_PATH not found" >&2
    exit 1
  fi
  echo "      ${BAK_ZIP_NAME}: $(du -h "$BAK_ZIP_PATH" | cut -f1)"
fi

# ---------------------------------------------------------------------------
# [3/3] Push each zip to Drive, verify, clean the RAID
# ---------------------------------------------------------------------------
echo "[3/3] Pushing to ${REMOTE}:${REMOTE_RESULTS_DIR}/"
if [[ "${#RCLONE_ROOT_FLAGS[@]}" -gt 0 && "$DRY_RUN" -eq 0 ]]; then
  echo "      (pinned to shared folder id: ${DRIVE_ROOT_FOLDER_ID})"
fi
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "      (dry-run — no upload, no RAID cleanup)"
fi

push_zip "$ZIP_PATH" "$RAID_RESULTS"
if [[ "$MERGE_BAK" -eq 1 ]]; then
  push_zip "$BAK_ZIP_PATH" "$RAID_BAK"
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "done (dry-run). Nothing uploaded; RAID staging kept for inspection:"
  echo "  ${RAID_STAGING_DIR}/"
  exit 0
fi

echo "done. Archived ${ZIP_NAME} to ${REMOTE}:${REMOTE_RESULTS_DIR}/${ZIP_NAME}"
if [[ "$MERGE_BAK" -eq 1 ]]; then
  echo "      archived ${BAK_ZIP_NAME} to ${REMOTE}:${REMOTE_RESULTS_DIR}/${BAK_ZIP_NAME}"
fi
if [[ "${#RCLONE_ROOT_FLAGS[@]}" -gt 0 ]]; then
  echo "List archives:  rclone lsf ${REMOTE}:${REMOTE_RESULTS_DIR}/ --drive-root-folder-id ${DRIVE_ROOT_FOLDER_ID}"
else
  echo "List archives:  rclone lsf ${REMOTE}:${REMOTE_RESULTS_DIR}/"
fi
echo "Pull latest:     ./sync_results_local.sh"
echo "Pull a specific: ./sync_results_local.sh ${ZIP_NAME}"
