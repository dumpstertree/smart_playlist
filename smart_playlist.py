#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
smart_playlist.py
-----------------
AI-powered smart playlist generator using Essentia Discogs-EffNet embeddings.

Usage:
    python smart_playlist.py [config.yaml] [--sync-only] [--upload-only] [-v]

Requires:
    pip install --pre essentia-tensorflow mutagen pyyaml numpy

Model download:
    https://essentia.upf.edu/models.html#discogs-effnet
    Get: discogs-effnet-bs64-1.pb
"""

import os
import sys
import yaml
import sqlite3
import hashlib
import logging
import argparse
import numpy as np
import urllib.request
import urllib.error
import json
from abc import ABC, abstractmethod
from pathlib import Path
from difflib import SequenceMatcher

try:
    from mutagen import File as MutagenFile
except ImportError:
    print("ERROR: mutagen is required:  pip install mutagen")
    sys.exit(1)

try:
    import essentia.standard as es
except ImportError:
    print("ERROR: essentia-tensorflow is required:  pip install --pre essentia-tensorflow")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def ensure_dir(path):
    """Create directory (and parents) if it does not exist."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULTS = {
    "library_path": ".",
    "db_path": "/app/embeddings.db",
    "model_path": "/models/discogs-effnet-bs64-1.pb",
    "output_m3u": "/app/playlist.m3u",
    "audio_formats": ["mp3", "flac"],
    "num_anchors": 5,
    "interpolation_count": 5,
    "interpolation_mode": "path",
    "allow_anchors_in_pool": False,
    "fuzzy_match_threshold": 0.6,
    "anchors": [],
    "exclude": [],
    # AzuraCast -- all optional, upload skipped if not set
    "azuracast": {
        "url": "",
        "api_key": "",
        "station_id": 0,
        "playlist_id": 0,
    },
}


def load_config(path):
    with open(path, encoding="utf-8") as f:
        user = yaml.safe_load(f) or {}

    # Deep-merge azuracast block so partial user config still works
    config = {**DEFAULTS, **user}
    az_defaults = dict(DEFAULTS["azuracast"])
    az_user = user.get("azuracast", {}) or {}
    config["azuracast"] = {**az_defaults, **az_user}

    # Normalise exclude to a set of lowercase stripped strings for matching
    config["_exclude_set"] = {s.lower().strip() for s in (config.get("exclude") or [])}

    if not config["anchors"]:
        raise ValueError("Config must contain at least one entry under 'anchors'.")
    if config["num_anchors"] < 2:
        raise ValueError("'num_anchors' must be at least 2.")
    if config["interpolation_count"] < 0:
        raise ValueError("'interpolation_count' cannot be negative.")

    return config


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
    path        TEXT    PRIMARY KEY,
    artist      TEXT    NOT NULL DEFAULT '',
    title       TEXT    NOT NULL DEFAULT '',
    mtime       REAL    NOT NULL,
    file_hash   TEXT    NOT NULL,
    embedding   BLOB
);
"""


def init_db(db_path):
    ensure_dir(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(SCHEMA)
    conn.commit()
    return conn


def db_get_all(conn):
    """Return {path: info_dict} for every track that has an embedding."""
    rows = conn.execute(
        "SELECT path, artist, title, mtime, file_hash, embedding "
        "FROM tracks WHERE embedding IS NOT NULL"
    ).fetchall()
    result = {}
    for path, artist, title, mtime, fhash, blob in rows:
        result[path] = {
            "artist": artist,
            "title": title,
            "mtime": mtime,
            "file_hash": fhash,
            "embedding": np.frombuffer(blob, dtype=np.float32).copy(),
        }
    return result


def db_get_index(conn):
    rows = conn.execute("SELECT path, mtime, file_hash FROM tracks").fetchall()
    return {r[0]: {"mtime": r[1], "file_hash": r[2]} for r in rows}


def db_upsert(conn, path, artist, title, mtime, fhash, embedding):
    blob = embedding.astype(np.float32).tobytes() if embedding is not None else None
    conn.execute(
        "INSERT OR REPLACE INTO tracks (path, artist, title, mtime, file_hash, embedding) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (path, artist, title, mtime, fhash, blob),
    )
    conn.commit()


def db_delete(conn, path):
    conn.execute("DELETE FROM tracks WHERE path = ?", (path,))
    conn.commit()


# ---------------------------------------------------------------------------
# File utilities
# ---------------------------------------------------------------------------

def quick_hash(path):
    """MD5 of first 64 KB -- fast change detection without reading the whole file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(65536))
    return h.hexdigest()


def read_tags(path):
    """Return (artist, title) from audio tags, falling back to filename."""
    try:
        tags = MutagenFile(path, easy=True)
        if tags is None:
            return ("", Path(path).stem)
        artist = str((tags.get("artist") or [""])[0])
        title = str((tags.get("title") or [""])[0]) or Path(path).stem
        return (artist, title)
    except Exception:
        return ("", Path(path).stem)


def scan_library(library_path, formats):
    exts = {".{}".format(fmt.lower()) for fmt in formats}
    found = []
    for root, _, files in os.walk(library_path):
        for fname in files:
            if Path(fname).suffix.lower() in exts:
                found.append(os.path.join(root, fname))
    return sorted(found)


# ---------------------------------------------------------------------------
# Exclusion helpers
# ---------------------------------------------------------------------------

def is_excluded(path, info, exclude_set):
    """
    Return True if this track matches any entry in the exclusion list.
    Matching is done against 'artist - title' and the bare filename,
    both lowercased and stripped.
    """
    if not exclude_set:
        return False
    candidate_full = "{} - {}".format(info["artist"], info["title"]).lower().strip()
    candidate_file = Path(path).stem.lower().strip()
    return candidate_full in exclude_set or candidate_file in exclude_set


def apply_exclusions(tracks, exclude_set):
    """Return a copy of tracks with excluded entries removed."""
    if not exclude_set:
        return tracks
    filtered = {p: t for p, t in tracks.items() if not is_excluded(p, t, exclude_set)}
    removed = len(tracks) - len(filtered)
    if removed:
        log.info("Excluded {} track(s) from the playlist pool.".format(removed))
    return filtered


# ---------------------------------------------------------------------------
# Essentia model
# ---------------------------------------------------------------------------

_MODEL_CACHE = {}


def load_model(model_path):
    if model_path not in _MODEL_CACHE:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "Essentia model not found at '{}'.\n"
                "Download discogs-effnet-bs64-1.pb from:\n"
                "  https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb".format(
                    model_path
                )
            )
        log.info("Loading Essentia model: {}".format(model_path))
        _MODEL_CACHE[model_path] = es.TensorflowPredictEffnetDiscogs(
            graphFilename=model_path,
            output="PartitionedCall:1",
        )
    return _MODEL_CACHE[model_path]


def compute_embedding(path, model):
    """
    Returns a 1-D mean-pooled embedding vector for the track, or None on failure.
    The EffNet model outputs (N_frames x 1280); averaging yields a single 1280-d vector.
    """
    try:
        audio = es.MonoLoader(filename=path, sampleRate=16000, resampleQuality=4)()
        frames = model(audio)
        return np.mean(frames, axis=0).astype(np.float32)
    except Exception as exc:
        log.warning("Embedding failed for {}: {}".format(os.path.basename(path), exc))
        return None


# ---------------------------------------------------------------------------
# Library sync
# ---------------------------------------------------------------------------

def sync_library(conn, config):
    """
    Scan library, add/update new or changed files, remove deleted files.
    Only loads the Essentia model if there is actually work to do.
    Returns the full up-to-date track dict (before exclusions are applied).
    """
    library_path = config["library_path"]
    formats = config["audio_formats"]
    model_path = config["model_path"]

    log.info("Scanning library: {}".format(library_path))
    disk_files = set(scan_library(library_path, formats))
    db_index = db_get_index(conn)
    db_files = set(db_index.keys())

    # Remove deleted files
    for path in db_files - disk_files:
        log.info("  Removing (deleted): {}".format(os.path.basename(path)))
        db_delete(conn, path)

    # Find files that need (re)analysis
    to_analyze = []
    for path in disk_files:
        mtime = os.path.getmtime(path)
        if path not in db_index:
            to_analyze.append(path)
        elif abs(mtime - db_index[path]["mtime"]) > 1.0:
            if quick_hash(path) != db_index[path]["file_hash"]:
                to_analyze.append(path)

    if to_analyze:
        log.info("Analysing {} new/changed track(s)...".format(len(to_analyze)))
        model = load_model(model_path)
        for i, path in enumerate(to_analyze, 1):
            log.info("  [{}/{}] {}".format(i, len(to_analyze), os.path.basename(path)))
            artist, title = read_tags(path)
            mtime = os.path.getmtime(path)
            fhash = quick_hash(path)
            emb = compute_embedding(path, model)
            db_upsert(conn, path, artist, title, mtime, fhash, emb)
    else:
        log.info("Library is up to date -- no analysis needed.")

    return db_get_all(conn)


# ---------------------------------------------------------------------------
# Fuzzy matching
# ---------------------------------------------------------------------------

def _fuzzy(a, b):
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def resolve_anchor(query, tracks, threshold, exclude):
    """Return the best-matching path for an 'Artist - Title' query, or None."""
    best_path = None
    best_score = 0.0
    for path, info in tracks.items():
        if path in exclude:
            continue
        candidate = "{} - {}".format(info["artist"], info["title"])
        score = _fuzzy(query, candidate)
        if score > best_score:
            best_score = score
            best_path = path
    return best_path if best_score >= threshold else None


def resolve_anchors(config, tracks):
    """
    Walk the anchor list from the config and resolve each entry.
    Unresolvable entries are skipped with a warning.
    Raises ValueError if fewer than num_anchors can be resolved.

    Note: there is no upper limit on num_anchors -- set it to however
    many you need. The only requirement is that enough entries exist
    in 'anchors' to satisfy the count after fuzzy matching.
    """
    num_wanted = config["num_anchors"]
    threshold = config.get("fuzzy_match_threshold", 0.6)
    resolved = []
    failed = []

    for query in config["anchors"]:
        path = resolve_anchor(query, tracks, threshold, exclude=set(resolved))
        if path:
            resolved.append(path)
            log.info("  Anchor OK  '{}'  ->  {} - {}".format(
                query, tracks[path]["artist"], tracks[path]["title"]
            ))
            if len(resolved) == num_wanted:
                break
        else:
            failed.append(query)
            log.warning("  Anchor MISS '{}' (no match above threshold {})".format(
                query, threshold
            ))

    if len(resolved) < num_wanted:
        raise ValueError(
            "Could not resolve enough anchors.\n"
            "  Needed : {}\n"
            "  Found  : {}\n"
            "  Failed : {}\n"
            "Add more entries to 'anchors' in your config, "
            "or lower 'fuzzy_match_threshold'.".format(num_wanted, len(resolved), failed)
        )
    return resolved


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------

def cosine_sim(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def nearest_to(target, pool, exclude, n=1):
    """Return the n paths from pool (minus exclude) closest to target."""
    scores = [
        (cosine_sim(target, info["embedding"]), path)
        for path, info in pool.items()
        if path not in exclude
    ]
    scores.sort(reverse=True)
    return [p for _, p in scores[:n]]


# ---------------------------------------------------------------------------
# Interpolation strategies
# ---------------------------------------------------------------------------

class InterpolationStrategy(ABC):
    """
    Base class for interpolation strategies.
    To add a new mode: subclass this, implement fill(), and register in STRATEGIES.
    """

    @abstractmethod
    def fill(self, anchor_a, anchor_b, count, tracks, pool, used):
        """
        Return a list of `count` track paths to insert between anchor_a and anchor_b.
        - tracks : full library dict (for embedding look-ups)
        - pool   : candidate tracks for interpolation (may exclude anchors)
        - used   : globally used paths -- do not pick any of these
        May return fewer than `count` if the pool is exhausted.
        """

    @staticmethod
    def _emb(path, tracks):
        return tracks[path]["embedding"]


class ChainStrategy(InterpolationStrategy):
    """
    Greedy nearest-neighbour chain.
    Each song is chosen as the closest unused track to the previous song,
    with no directional pull toward anchor_b.
    Best for: smooth organic flow from A where you don't care about arriving at B.
    """

    def fill(self, anchor_a, anchor_b, count, tracks, pool, used):
        result = []
        prev_emb = self._emb(anchor_a, tracks)
        for _ in range(count):
            picks = nearest_to(prev_emb, pool, used | set(result))
            if not picks:
                break
            chosen = picks[0]
            result.append(chosen)
            prev_emb = self._emb(chosen, tracks)
        return result


class PathStrategy(InterpolationStrategy):
    """
    Linear embedding interpolation.
    Evenly spaces `count` target points along the straight line in embedding
    space from A to B, then picks the closest unused track to each point.
    Best for: intentional audible journey from A's sound toward B's sound.
    """

    def fill(self, anchor_a, anchor_b, count, tracks, pool, used):
        emb_a = self._emb(anchor_a, tracks)
        emb_b = self._emb(anchor_b, tracks)
        result = []
        local_used = set()

        for i in range(1, count + 1):
            t = i / (count + 1)
            target = (1.0 - t) * emb_a + t * emb_b
            picks = nearest_to(target, pool, used | local_used)
            if not picks:
                break
            chosen = picks[0]
            result.append(chosen)
            local_used.add(chosen)

        return result


class MidpointStrategy(InterpolationStrategy):
    """
    Midpoint anchor + symmetric path fill.
    Finds the best track for the midpoint between A and B, then fills
    the first half as a path from A to midpoint and second half from midpoint to B.
    Best for: playlists with a clear peak or pivot in the middle of each segment.
    """

    def fill(self, anchor_a, anchor_b, count, tracks, pool, used):
        if count == 0:
            return []

        emb_a = self._emb(anchor_a, tracks)
        emb_b = self._emb(anchor_b, tracks)
        mid_emb = (emb_a + emb_b) / 2.0
        local_used = set()

        mid_picks = nearest_to(mid_emb, pool, used | local_used)
        if not mid_picks:
            return []
        mid_track = mid_picks[0]
        local_used.add(mid_track)

        remaining = count - 1
        half_before = remaining // 2
        half_after = remaining - half_before

        before = []
        for i in range(1, half_before + 1):
            t = i / (half_before + 1)
            target = (1.0 - t) * emb_a + t * self._emb(mid_track, tracks)
            picks = nearest_to(target, pool, used | local_used | set(before))
            if not picks:
                break
            before.append(picks[0])

        after = []
        for i in range(1, half_after + 1):
            t = i / (half_after + 1)
            target = (1.0 - t) * self._emb(mid_track, tracks) + t * emb_b
            picks = nearest_to(
                target, pool, used | local_used | set(before) | set(after)
            )
            if not picks:
                break
            after.append(picks[0])

        return before + [mid_track] + after


# Registry -- add new strategies here without touching any other code
STRATEGIES = {
    "chain":    ChainStrategy(),
    "path":     PathStrategy(),
    "midpoint": MidpointStrategy(),
}


# ---------------------------------------------------------------------------
# Playlist builder
# ---------------------------------------------------------------------------

def build_playlist(config, tracks):
    mode = config["interpolation_mode"]
    count = config["interpolation_count"]

    if mode not in STRATEGIES:
        raise ValueError(
            "Unknown interpolation_mode '{}'.\n"
            "Valid modes: {}".format(mode, sorted(STRATEGIES.keys()))
        )
    strategy = STRATEGIES[mode]

    # Apply exclusions before building
    exclude_set = config.get("_exclude_set", set())
    tracks = apply_exclusions(tracks, exclude_set)

    log.info("Resolving anchors  (mode={}, interpolation_count={})".format(mode, count))
    anchors = resolve_anchors(config, tracks)

    anchor_set = set(anchors)
    pool = (
        tracks
        if config.get("allow_anchors_in_pool", False)
        else {p: t for p, t in tracks.items() if p not in anchor_set}
    )

    used = set(anchors)
    playlist = []

    for i in range(len(anchors) - 1):
        a, b = anchors[i], anchors[i + 1]
        playlist.append(a)

        log.info("  Segment {}: {} - {}  ->  {} - {}".format(
            i + 1,
            tracks[a]["artist"], tracks[a]["title"],
            tracks[b]["artist"], tracks[b]["title"],
        ))

        interp = strategy.fill(a, b, count, tracks, pool, used)

        if len(interp) < count:
            log.warning("    Only found {}/{} interpolation tracks "
                        "(pool may be exhausted).".format(len(interp), count))

        used.update(interp)
        playlist.extend(interp)

    playlist.append(anchors[-1])

    log.info("Playlist complete: {} tracks total.".format(len(playlist)))
    return playlist


# ---------------------------------------------------------------------------
# M3U output
# ---------------------------------------------------------------------------

def write_m3u(playlist, tracks, output_path):
    ensure_dir(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("#EXTM3U\n")
        for path in playlist:
            info = tracks.get(path, {})
            artist = info.get("artist", "")
            title = info.get("title", os.path.basename(path))
            label = "{} - {}".format(artist, title) if artist else title
            f.write("#EXTINF:-1,{}\n{}\n".format(label, path))
    log.info("M3U written -> {}  ({} tracks)".format(output_path, len(playlist)))


# ---------------------------------------------------------------------------
# AzuraCast upload
# ---------------------------------------------------------------------------

def azuracast_upload(m3u_path, az_config):
    """
    Import the generated M3U into an AzuraCast playlist via the REST API.

    Requires in config.azuracast:
        url         -- base URL, e.g. http://192.168.1.10 or https://radio.example.com
        api_key     -- AzuraCast API key (Admin > API Keys)
        station_id  -- numeric station ID
        playlist_id -- numeric playlist ID to overwrite
    """
    url = az_config.get("url", "").rstrip("/")
    api_key = az_config.get("api_key", "")
    station_id = az_config.get("station_id", 0)
    playlist_id = az_config.get("playlist_id", 0)

    if not all([url, api_key, station_id, playlist_id]):
        log.info("AzuraCast config incomplete -- skipping upload.")
        return

    endpoint = "{}/api/station/{}/playlist/{}/import".format(
        url, station_id, playlist_id
    )

    log.info("Uploading playlist to AzuraCast: {}".format(endpoint))

    with open(m3u_path, "rb") as f:
        m3u_bytes = f.read()

    # Build a minimal multipart/form-data body manually to avoid needing requests lib
    boundary = "----PlaylistBoundary"
    body = (
        "--{}\r\n"
        "Content-Disposition: form-data; name=\"playlist_file\"; filename=\"playlist.m3u\"\r\n"
        "Content-Type: audio/x-mpegurl\r\n"
        "\r\n"
    ).format(boundary).encode("utf-8") + m3u_bytes + "\r\n--{}--\r\n".format(boundary).encode("utf-8")

    req = urllib.request.Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "X-API-Key": api_key,
            "Content-Type": "multipart/form-data; boundary={}".format(boundary),
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
            try:
                result = json.loads(raw)
                found = result.get("files_found", "?")
                success = result.get("files_success", "?")
                log.info("AzuraCast upload complete: {}/{} tracks imported.".format(
                    success, found
                ))
            except json.JSONDecodeError:
                log.info("AzuraCast upload complete (raw response): {}".format(raw[:200]))
    except urllib.error.HTTPError as e:
        log.error("AzuraCast HTTP error {}: {}".format(e.code, e.read().decode("utf-8", errors="replace")))
    except urllib.error.URLError as e:
        log.error("AzuraCast connection error: {}".format(e.reason))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Smart playlist generator -- Essentia Discogs-EffNet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    parser.add_argument(
        "--sync-only",
        action="store_true",
        help="Sync the library embeddings and exit without generating a playlist.",
    )
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="Skip generation and just upload the existing M3U to AzuraCast.",
    )
    parser.add_argument(
        "--list-modes",
        action="store_true",
        help="Print available interpolation modes and exit.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.list_modes:
        print("Available interpolation modes:")
        for name, strat in STRATEGIES.items():
            print("  {:12s}  {}".format(
                name, strat.__class__.__doc__.strip().splitlines()[0]
            ))
        sys.exit(0)

    if not os.path.exists(args.config):
        log.error("Config file not found: {}".format(args.config))
        sys.exit(1)

    try:
        config = load_config(args.config)
    except (ValueError, yaml.YAMLError) as exc:
        log.error("Config error: {}".format(exc))
        sys.exit(1)

    # --upload-only: skip analysis and generation, just push existing M3U
    if args.upload_only:
        m3u = config["output_m3u"]
        if not os.path.exists(m3u):
            log.error("M3U not found at '{}' -- run without --upload-only first.".format(m3u))
            sys.exit(1)
        azuracast_upload(m3u, config["azuracast"])
        return

    conn = init_db(config["db_path"])
    tracks = sync_library(conn, config)

    if not tracks:
        log.error("No tracks with embeddings found. Check library_path and audio_formats.")
        sys.exit(1)

    log.info("Library ready: {} tracks with embeddings.".format(len(tracks)))

    if args.sync_only:
        log.info("--sync-only: done.")
        return

    try:
        playlist = build_playlist(config, tracks)
    except ValueError as exc:
        log.error(str(exc))
        sys.exit(1)

    write_m3u(playlist, tracks, config["output_m3u"])
    azuracast_upload(config["output_m3u"], config["azuracast"])


if __name__ == "__main__":
    main()
