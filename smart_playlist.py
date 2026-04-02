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
    https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb
"""

import os
import sys
import yaml
import sqlite3
import hashlib
import logging
import argparse
import random
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
    "interpolation_mode": "diverse",
    "allow_anchors_in_pool": False,
    "fuzzy_match_threshold": 0.6,
    "loop": False,
    # Anchor sources -- all optional, combined into one pool then randomly sampled
    "anchor_songs": [],
    "anchor_albums": [],
    "anchor_artists": [],
    "anchor_genres": [],
    # Exclusion sources -- all optional, union of all matches is removed from pool
    "exclude_songs": [],
    "exclude_albums": [],
    "exclude_artists": [],
    "exclude_genres": [],
    # Diverse mode tuning
    "diverse_artist_window": 3,
    "diverse_artist_penalty": 0.35,
    "diverse_curve_power": 0.55,
    "diverse_candidates": 20,
    # AzuraCast -- all optional, upload skipped if not configured
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

    config = {**DEFAULTS, **user}

    # Deep-merge azuracast block
    az_defaults = dict(DEFAULTS["azuracast"])
    az_user = user.get("azuracast", {}) or {}
    config["azuracast"] = {**az_defaults, **az_user}

    has_anchors = any(
        config.get(k) for k in
        ("anchor_songs", "anchor_albums", "anchor_artists", "anchor_genres")
    )
    if not has_anchors:
        raise ValueError(
            "Config must contain at least one anchor source:\n"
            "  anchor_songs, anchor_albums, anchor_artists, or anchor_genres"
        )
    if config["num_anchors"] < 2:
        raise ValueError("'num_anchors' must be at least 2.")
    if config["interpolation_count"] < 0:
        raise ValueError("'interpolation_count' cannot be negative.")

    return config


# ---------------------------------------------------------------------------
# Database  (schema v2 adds album + genre columns)
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
    path        TEXT    PRIMARY KEY,
    artist      TEXT    NOT NULL DEFAULT '',
    title       TEXT    NOT NULL DEFAULT '',
    album       TEXT    NOT NULL DEFAULT '',
    genre       TEXT    NOT NULL DEFAULT '',
    mtime       REAL    NOT NULL,
    file_hash   TEXT    NOT NULL,
    embedding   BLOB
);
"""

MIGRATION = """
ALTER TABLE tracks ADD COLUMN album TEXT NOT NULL DEFAULT '';
ALTER TABLE tracks ADD COLUMN genre TEXT NOT NULL DEFAULT '';
"""


def init_db(db_path):
    ensure_dir(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(SCHEMA)
    conn.commit()

    # Migrate existing DBs that pre-date album/genre columns
    cols = {row[1] for row in conn.execute("PRAGMA table_info(tracks)")}
    if "album" not in cols:
        conn.execute("ALTER TABLE tracks ADD COLUMN album TEXT NOT NULL DEFAULT ''")
        log.info("DB migrated: added 'album' column.")
    if "genre" not in cols:
        conn.execute("ALTER TABLE tracks ADD COLUMN genre TEXT NOT NULL DEFAULT ''")
        log.info("DB migrated: added 'genre' column.")
    conn.commit()
    return conn


def db_get_all(conn):
    rows = conn.execute(
        "SELECT path, artist, title, album, genre, mtime, file_hash, embedding "
        "FROM tracks WHERE embedding IS NOT NULL"
    ).fetchall()
    result = {}
    for path, artist, title, album, genre, mtime, fhash, blob in rows:
        result[path] = {
            "artist": artist,
            "title":  title,
            "album":  album,
            "genre":  genre,
            "mtime":  mtime,
            "file_hash": fhash,
            "embedding": np.frombuffer(blob, dtype=np.float32).copy(),
        }
    return result


def db_get_index(conn):
    rows = conn.execute("SELECT path, mtime, file_hash FROM tracks").fetchall()
    return {r[0]: {"mtime": r[1], "file_hash": r[2]} for r in rows}


def db_upsert(conn, path, artist, title, album, genre, mtime, fhash, embedding):
    blob = embedding.astype(np.float32).tobytes() if embedding is not None else None
    conn.execute(
        "INSERT OR REPLACE INTO tracks "
        "(path, artist, title, album, genre, mtime, file_hash, embedding) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (path, artist, title, album, genre, mtime, fhash, blob),
    )
    conn.commit()


def db_delete(conn, path):
    conn.execute("DELETE FROM tracks WHERE path = ?", (path,))
    conn.commit()


# ---------------------------------------------------------------------------
# File utilities
# ---------------------------------------------------------------------------

def quick_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(65536))
    return h.hexdigest()


def read_tags(path):
    """Return (artist, title, album, genre) from audio tags."""
    try:
        tags = MutagenFile(path, easy=True)
        if tags is None:
            return ("", Path(path).stem, "", "")
        artist = str((tags.get("artist") or [""])[0])
        title  = str((tags.get("title")  or [""])[0]) or Path(path).stem
        album  = str((tags.get("album")  or [""])[0])
        genre  = str((tags.get("genre")  or [""])[0])
        return (artist, title, album, genre)
    except Exception:
        return ("", Path(path).stem, "", "")


def scan_library(library_path, formats):
    exts = {".{}".format(fmt.lower()) for fmt in formats}
    found = []
    for root, _, files in os.walk(library_path):
        for fname in files:
            if Path(fname).suffix.lower() in exts:
                found.append(os.path.join(root, fname))
    return sorted(found)


# ---------------------------------------------------------------------------
# Fuzzy matching helpers
# ---------------------------------------------------------------------------

def _fuzzy(a, b):
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def _best_fuzzy_score(query, candidates):
    """Return the highest fuzzy score between query and any string in candidates."""
    if not candidates:
        return 0.0
    return max(_fuzzy(query, c) for c in candidates if c)


def matches_song(query, info, threshold):
    """Match 'Artist - Title' against a track's tags."""
    candidate = "{} - {}".format(info["artist"], info["title"])
    return _fuzzy(query, candidate) >= threshold


def matches_artist(query, info, threshold):
    return bool(info["artist"]) and _fuzzy(query, info["artist"]) >= threshold


def matches_album(query, info, threshold):
    """Match 'Artist - Album' or plain album name."""
    if not info["album"]:
        return False
    full = "{} - {}".format(info["artist"], info["album"])
    return _fuzzy(query, full) >= threshold or _fuzzy(query, info["album"]) >= threshold


def matches_genre(query, info, threshold):
    """Match against the genre tag; supports comma/slash separated multi-genre fields."""
    raw = info.get("genre", "")
    if not raw:
        return False
    parts = [g.strip() for g in raw.replace("/", ",").split(",") if g.strip()]
    return _best_fuzzy_score(query, parts) >= threshold


# ---------------------------------------------------------------------------
# Anchor pool builder
# ---------------------------------------------------------------------------

def build_anchor_pool(config, tracks):
    """
    Compile the full set of candidate anchor tracks from all anchor_* lists.
    Returns a list of paths (may contain duplicates if a track matches multiple
    criteria -- deduplicated before returning).
    """
    threshold = config.get("fuzzy_match_threshold", 0.6)
    pool = set()

    # anchor_songs
    for query in (config.get("anchor_songs") or []):
        matched = [p for p, info in tracks.items() if matches_song(query, info, threshold)]
        if matched:
            pool.update(matched)
            log.info("  anchor_song '{}' -> {} track(s)".format(query, len(matched)))
        else:
            log.warning("  anchor_song '{}' -> no match".format(query))

    # anchor_artists
    for query in (config.get("anchor_artists") or []):
        matched = [p for p, info in tracks.items() if matches_artist(query, info, threshold)]
        if matched:
            pool.update(matched)
            log.info("  anchor_artist '{}' -> {} track(s)".format(query, len(matched)))
        else:
            log.warning("  anchor_artist '{}' -> no match".format(query))

    # anchor_albums
    for query in (config.get("anchor_albums") or []):
        matched = [p for p, info in tracks.items() if matches_album(query, info, threshold)]
        if matched:
            pool.update(matched)
            log.info("  anchor_album '{}' -> {} track(s)".format(query, len(matched)))
        else:
            log.warning("  anchor_album '{}' -> no match".format(query))

    # anchor_genres
    for query in (config.get("anchor_genres") or []):
        matched = [p for p, info in tracks.items() if matches_genre(query, info, threshold)]
        if matched:
            pool.update(matched)
            log.info("  anchor_genre '{}' -> {} track(s)".format(query, len(matched)))
        else:
            log.warning("  anchor_genre '{}' -> no match".format(query))

    return list(pool)


def resolve_anchors(config, tracks):
    """
    Build the full anchor candidate pool from all anchor_* sources,
    then randomly sample num_anchors from it.
    Raises ValueError if the pool is too small to satisfy num_anchors.
    """
    num_wanted = config["num_anchors"]

    log.info("Building anchor pool...")
    pool = build_anchor_pool(config, tracks)

    if len(pool) < num_wanted:
        raise ValueError(
            "Anchor pool too small.\n"
            "  Needed : {}\n"
            "  Found  : {}\n"
            "Add more entries to anchor_songs / anchor_artists / "
            "anchor_albums / anchor_genres.".format(num_wanted, len(pool))
        )

    chosen = random.sample(pool, num_wanted)
    log.info("Anchor pool: {} candidates, {} randomly selected.".format(
        len(pool), num_wanted
    ))
    for path in chosen:
        info = tracks[path]
        log.info("  Anchor: {} - {} [{}]".format(
            info["artist"], info["title"], info["album"]
        ))
    return chosen


# ---------------------------------------------------------------------------
# Exclusion builder
# ---------------------------------------------------------------------------

def build_exclude_set(config, tracks):
    """
    Compile the full set of paths to exclude from the interpolation pool.
    Returns a set of paths.
    """
    threshold = config.get("fuzzy_match_threshold", 0.6)
    excluded = set()

    for query in (config.get("exclude_songs") or []):
        matched = [p for p, info in tracks.items() if matches_song(query, info, threshold)]
        if matched:
            excluded.update(matched)
            log.info("  exclude_song '{}' -> {} track(s)".format(query, len(matched)))
        else:
            log.warning("  exclude_song '{}' -> no match".format(query))

    for query in (config.get("exclude_artists") or []):
        matched = [p for p, info in tracks.items() if matches_artist(query, info, threshold)]
        if matched:
            excluded.update(matched)
            log.info("  exclude_artist '{}' -> {} track(s)".format(query, len(matched)))
        else:
            log.warning("  exclude_artist '{}' -> no match".format(query))

    for query in (config.get("exclude_albums") or []):
        matched = [p for p, info in tracks.items() if matches_album(query, info, threshold)]
        if matched:
            excluded.update(matched)
            log.info("  exclude_album '{}' -> {} track(s)".format(query, len(matched)))
        else:
            log.warning("  exclude_album '{}' -> no match".format(query))

    for query in (config.get("exclude_genres") or []):
        matched = [p for p, info in tracks.items() if matches_genre(query, info, threshold)]
        if matched:
            excluded.update(matched)
            log.info("  exclude_genre '{}' -> {} track(s)".format(query, len(matched)))
        else:
            log.warning("  exclude_genre '{}' -> no match".format(query))

    if excluded:
        log.info("Total excluded from interpolation pool: {} track(s).".format(len(excluded)))
    return excluded


# ---------------------------------------------------------------------------
# Essentia model
# ---------------------------------------------------------------------------

_MODEL_CACHE = {}


def load_model(model_path):
    if model_path not in _MODEL_CACHE:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "Essentia model not found at '{}'.\n"
                "Download from:\n"
                "  https://essentia.upf.edu/models/feature-extractors/"
                "discogs-effnet/discogs-effnet-bs64-1.pb".format(model_path)
            )
        log.info("Loading Essentia model: {}".format(model_path))
        _MODEL_CACHE[model_path] = es.TensorflowPredictEffnetDiscogs(
            graphFilename=model_path,
            output="PartitionedCall:1",
        )
    return _MODEL_CACHE[model_path]


def compute_embedding(path, model):
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
    library_path = config["library_path"]
    formats      = config["audio_formats"]
    model_path   = config["model_path"]

    log.info("Scanning library: {}".format(library_path))
    disk_files = set(scan_library(library_path, formats))
    db_index   = db_get_index(conn)
    db_files   = set(db_index.keys())

    for path in db_files - disk_files:
        log.info("  Removing (deleted): {}".format(os.path.basename(path)))
        db_delete(conn, path)

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
            artist, title, album, genre = read_tags(path)
            mtime  = os.path.getmtime(path)
            fhash  = quick_hash(path)
            emb    = compute_embedding(path, model)
            db_upsert(conn, path, artist, title, album, genre, mtime, fhash, emb)
    else:
        log.info("Library is up to date -- no analysis needed.")

    return db_get_all(conn)


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------

def cosine_sim(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def nearest_to(target, pool, exclude, n=1):
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
    @abstractmethod
    def fill(self, anchor_a, anchor_b, count, tracks, pool, used, config,
             recent_artists=None):
        pass

    @staticmethod
    def _emb(path, tracks):
        return tracks[path]["embedding"]


class ChainStrategy(InterpolationStrategy):
    """
    Greedy nearest-neighbour chain.
    Each song is chosen as the closest unused track to the previous song,
    with no directional pull toward anchor_b.
    """

    def fill(self, anchor_a, anchor_b, count, tracks, pool, used, config,
             recent_artists=None):
        result = []
        prev_emb = self._emb(anchor_a, tracks)
        for _ in range(count):
            picks = nearest_to(prev_emb, pool, used | set(result))
            if not picks:
                break
            chosen = picks[0]
            result.append(chosen)
            prev_emb = self._emb(chosen, tracks)
            if recent_artists is not None:
                recent_artists.append(tracks[chosen]["artist"])
        return result


class PathStrategy(InterpolationStrategy):
    """
    Linear embedding interpolation.
    Evenly spaces target points along the line from A to B in embedding space.
    """

    def fill(self, anchor_a, anchor_b, count, tracks, pool, used, config,
             recent_artists=None):
        emb_a    = self._emb(anchor_a, tracks)
        emb_b    = self._emb(anchor_b, tracks)
        result   = []
        local_used = set()

        for i in range(1, count + 1):
            t      = i / (count + 1)
            target = (1.0 - t) * emb_a + t * emb_b
            picks  = nearest_to(target, pool, used | local_used)
            if not picks:
                break
            chosen = picks[0]
            result.append(chosen)
            local_used.add(chosen)
            if recent_artists is not None:
                recent_artists.append(tracks[chosen]["artist"])
        return result


class MidpointStrategy(InterpolationStrategy):
    """
    Midpoint anchor + symmetric path fill.
    Finds the best midpoint track then path-fills each half independently.
    """

    def fill(self, anchor_a, anchor_b, count, tracks, pool, used, config,
             recent_artists=None):
        if count == 0:
            return []

        emb_a    = self._emb(anchor_a, tracks)
        emb_b    = self._emb(anchor_b, tracks)
        mid_emb  = (emb_a + emb_b) / 2.0
        local_used = set()

        mid_picks = nearest_to(mid_emb, pool, used | local_used)
        if not mid_picks:
            return []
        mid_track = mid_picks[0]
        local_used.add(mid_track)

        remaining   = count - 1
        half_before = remaining // 2
        half_after  = remaining - half_before

        before = []
        for i in range(1, half_before + 1):
            t      = i / (half_before + 1)
            target = (1.0 - t) * emb_a + t * self._emb(mid_track, tracks)
            picks  = nearest_to(target, pool, used | local_used | set(before))
            if not picks:
                break
            before.append(picks[0])

        after = []
        for i in range(1, half_after + 1):
            t      = i / (half_after + 1)
            target = (1.0 - t) * self._emb(mid_track, tracks) + t * emb_b
            picks  = nearest_to(target, pool, used | local_used | set(before) | set(after))
            if not picks:
                break
            after.append(picks[0])

        result = before + [mid_track] + after
        if recent_artists is not None:
            for p in result:
                recent_artists.append(tracks[p]["artist"])
        return result


class DiverseStrategy(InterpolationStrategy):
    """
    Curved traversal + artist penalty. Recommended default.
    Moves away from anchor A faster (curve_power < 1.0) and penalises
    recently played artists to break up same-artist clustering.
    """

    def fill(self, anchor_a, anchor_b, count, tracks, pool, used, config,
             recent_artists=None):
        emb_a    = self._emb(anchor_a, tracks)
        emb_b    = self._emb(anchor_b, tracks)

        curve_power    = float(config.get("diverse_curve_power",    0.55))
        n_candidates   = int(config.get("diverse_candidates",       20))
        artist_window  = int(config.get("diverse_artist_window",    3))
        artist_penalty = float(config.get("diverse_artist_penalty", 0.35))

        if recent_artists is None:
            recent_artists = []

        recent_artists.append(tracks[anchor_a]["artist"])

        result     = []
        local_used = set()

        for i in range(1, count + 1):
            t_linear = i / (count + 1)
            t        = t_linear ** curve_power
            target   = (1.0 - t) * emb_a + t * emb_b

            candidates = nearest_to(target, pool, used | local_used, n=n_candidates)
            if not candidates:
                break

            window = recent_artists[-artist_window:] if artist_window > 0 else []

            def score(path):
                raw    = cosine_sim(target, tracks[path]["embedding"])
                artist = tracks[path]["artist"].lower().strip()
                hits   = sum(1 for a in window if a.lower().strip() == artist)
                return raw - (artist_penalty * hits)

            best = max(candidates, key=score)
            result.append(best)
            local_used.add(best)
            recent_artists.append(tracks[best]["artist"])

        return result


STRATEGIES = {
    "chain":    ChainStrategy(),
    "path":     PathStrategy(),
    "midpoint": MidpointStrategy(),
    "diverse":  DiverseStrategy(),
}


# ---------------------------------------------------------------------------
# Playlist builder
# ---------------------------------------------------------------------------

def build_playlist(config, tracks):
    mode  = config["interpolation_mode"]
    count = config["interpolation_count"]
    loop  = config.get("loop", False)

    if mode not in STRATEGIES:
        raise ValueError(
            "Unknown interpolation_mode '{}'.\n"
            "Valid modes: {}".format(mode, sorted(STRATEGIES.keys()))
        )
    strategy = STRATEGIES[mode]

    log.info("Building exclusion set...")
    exclude_set = build_exclude_set(config, tracks)

    # Remove excluded tracks from the pool available for interpolation.
    # Anchors may still come from excluded tracks if they were matched via
    # anchor_songs/artists/albums/genres -- exclusions only affect the fill pool.
    fill_pool = {p: t for p, t in tracks.items() if p not in exclude_set}

    log.info("Resolving anchors  (mode={}, interpolation_count={}, loop={})".format(
        mode, count, loop
    ))
    anchors = resolve_anchors(config, tracks)

    anchor_set = set(anchors)
    pool = (
        fill_pool
        if config.get("allow_anchors_in_pool", False)
        else {p: t for p, t in fill_pool.items() if p not in anchor_set}
    )

    used           = set(anchors)
    playlist       = []
    recent_artists = []

    for i in range(len(anchors) - 1):
        a, b = anchors[i], anchors[i + 1]
        playlist.append(a)
        recent_artists.append(tracks[a]["artist"])

        log.info("  Segment {}: {} - {}  ->  {} - {}".format(
            i + 1,
            tracks[a]["artist"], tracks[a]["title"],
            tracks[b]["artist"], tracks[b]["title"],
        ))

        interp = strategy.fill(a, b, count, tracks, pool, used, config, recent_artists)

        if len(interp) < count:
            log.warning("    Only found {}/{} interpolation tracks.".format(
                len(interp), count
            ))

        used.update(interp)
        playlist.extend(interp)

    playlist.append(anchors[-1])
    recent_artists.append(tracks[anchors[-1]]["artist"])

    if loop and len(anchors) >= 2:
        a = anchors[-1]
        b = anchors[0]
        log.info("  Loop segment: {} - {}  ->  {} - {}".format(
            tracks[a]["artist"], tracks[a]["title"],
            tracks[b]["artist"], tracks[b]["title"],
        ))
        loop_interp = strategy.fill(
            a, b, count, tracks, pool, used, config, recent_artists
        )
        if len(loop_interp) < count:
            log.warning("    Loop segment: only found {}/{} interpolation tracks.".format(
                len(loop_interp), count
            ))
        used.update(loop_interp)
        playlist.extend(loop_interp)
        log.info("  Loop bridge: {} track(s) added.".format(len(loop_interp)))

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
            info   = tracks.get(path, {})
            artist = info.get("artist", "")
            title  = info.get("title", os.path.basename(path))
            label  = "{} - {}".format(artist, title) if artist else title
            f.write("#EXTINF:-1,{}\n{}\n".format(label, path))
    log.info("M3U written -> {}  ({} tracks)".format(output_path, len(playlist)))


# ---------------------------------------------------------------------------
# AzuraCast upload
# ---------------------------------------------------------------------------

def _az_request(endpoint, api_key, method="GET", data=None, content_type=None):
    """Make a request to the AzuraCast API, return (status_code, body_str)."""
    headers = {"X-API-Key": api_key}
    if content_type:
        headers["Content-Type"] = content_type
    req = urllib.request.Request(endpoint, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as e:
        raise ConnectionError("AzuraCast connection error: {}".format(e.reason))
 
 
def azuracast_clear_playlist(base_url, api_key, station_id, playlist_id):
    """
    Clear all tracks from a playlist using the AzuraCast batch files API.
 
    Strategy:
      1. Paginate through all station files that belong to this playlist.
      2. Collect their internal file IDs.
      3. POST a single batch request that sets their playlists to empty,
         removing them from this playlist in one call.
 
    This is how the AzuraCast web UI clears a playlist internally.
    """
    log.info("Fetching current playlist contents...")
 
    file_ids = []
    page = 1
    per_page = 100
 
    while True:
        endpoint = "{}/api/station/{}/files?playlist[]={}&per_page={}&page={}".format(
            base_url, station_id, playlist_id, per_page, page
        )
        status, body = _az_request(endpoint, api_key)
 
        if status != 200:
            log.warning(
                "Could not fetch playlist files (HTTP {}). "
                "Skipping clear -- playlist may have duplicates after import.".format(status)
            )
            return
 
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            log.warning("Could not parse files response. Skipping clear.")
            return
 
        # AzuraCast paginates as {"rows": [...], "total": N} or just a plain list
        if isinstance(data, dict):
            rows = data.get("rows", [])
            total = data.get("total", 0)
        else:
            rows = data
            total = len(rows)
 
        for f in rows:
            fid = f.get("id")
            if fid is not None:
                file_ids.append(fid)
 
        if len(file_ids) >= total or len(rows) < per_page:
            break
        page += 1
 
    if not file_ids:
        log.info("Playlist is already empty.")
        return
 
    log.info("Clearing {} track(s) from playlist via batch API...".format(len(file_ids)))
 
    # Batch update: set playlists to [] for all files currently in this playlist.
    # AzuraCast interprets an empty playlists list as "remove from all playlists".
    batch_payload = json.dumps({
        "do": "playlist",
        "files": file_ids,
        "playlists": []
    }).encode("utf-8")
 
    batch_endpoint = "{}/api/station/{}/files/batch".format(base_url, station_id)
    status, body = _az_request(
        batch_endpoint, api_key,
        method="POST", data=batch_payload,
        content_type="application/json"
    )
 
    if status in (200, 204):
        log.info("Playlist cleared successfully.")
    else:
        log.warning(
            "Batch clear returned HTTP {}. "
            "Playlist may still contain old tracks. Response: {}".format(status, body[:200])
        )
 
 
def azuracast_upload(m3u_path, az_config):
    """
    Overwrite an AzuraCast playlist with the generated M3U.
    Clears existing contents first via the batch files API, then imports.
    """
    url         = az_config.get("url", "").rstrip("/")
    api_key     = az_config.get("api_key", "")
    station_id  = az_config.get("station_id", 0)
    playlist_id = az_config.get("playlist_id", 0)
 
    if not all([url, api_key, station_id, playlist_id]):
        log.info("AzuraCast config incomplete -- skipping upload.")
        return
 
    # Step 1: clear existing playlist contents
    azuracast_clear_playlist(url, api_key, station_id, playlist_id)
 
    # Step 2: import new M3U
    endpoint = "{}/api/station/{}/playlist/{}/import".format(
        url, station_id, playlist_id
    )
    log.info("Importing playlist to AzuraCast...")
 
    with open(m3u_path, "rb") as f:
        m3u_bytes = f.read()
 
    boundary = "----PlaylistBoundary"
    body = (
        "--{}\r\nContent-Disposition: form-data; name=\"playlist_file\";"
        " filename=\"playlist.m3u\"\r\nContent-Type: audio/x-mpegurl\r\n\r\n"
    ).format(boundary).encode("utf-8") \
        + m3u_bytes \
        + "\r\n--{}--\r\n".format(boundary).encode("utf-8")
 
    try:
        status, resp_body = _az_request(
            endpoint, api_key, method="POST", data=body,
            content_type="multipart/form-data; boundary={}".format(boundary)
        )
        try:
            result  = json.loads(resp_body)
            found   = result.get("files_found", "?")
            success = result.get("files_success", "?")
            log.info("AzuraCast import complete: {}/{} tracks matched.".format(
                success, found
            ))
        except json.JSONDecodeError:
            log.info("AzuraCast import response (HTTP {}): {}".format(
                status, resp_body[:200]
            ))
    except ConnectionError as e:
        log.error(str(e))

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
        "config", nargs="?", default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    parser.add_argument("--sync-only",   action="store_true",
                        help="Sync embeddings then exit.")
    parser.add_argument("--upload-only", action="store_true",
                        help="Upload existing M3U to AzuraCast without regenerating.")
    parser.add_argument("--list-modes",  action="store_true",
                        help="Print available interpolation modes and exit.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable debug logging.")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.list_modes:
        print("Available interpolation modes:")
        for name, strat in STRATEGIES.items():
            first = strat.__class__.__doc__.strip().splitlines()[0]
            print("  {:12s}  {}".format(name, first))
        sys.exit(0)

    if not os.path.exists(args.config):
        log.error("Config file not found: {}".format(args.config))
        sys.exit(1)

    try:
        config = load_config(args.config)
    except (ValueError, yaml.YAMLError) as exc:
        log.error("Config error: {}".format(exc))
        sys.exit(1)

    if args.upload_only:
        m3u = config["output_m3u"]
        if not os.path.exists(m3u):
            log.error("M3U not found at '{}'. Run without --upload-only first.".format(m3u))
            sys.exit(1)
        azuracast_upload(m3u, config["azuracast"])
        return

    conn   = init_db(config["db_path"])
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
