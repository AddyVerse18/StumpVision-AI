"""
store.py
========
Storage and persistence logic for delivery labels, clip reviewed status, and manifest views.
Supports atomic saves after every user action and blind relabeling mode.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd

from stumpvision.schemas import (
    SCHEMA_VERSION,
    LABELS_COLUMNS,
    CLIP_STATUS_COLUMNS,
    VALID_KINDS,
    VALID_VIEWS,
    validate_labels_df,
    validate_clip_status_df,
)


class LabelStore:
    def __init__(
        self,
        labels_path: Path,
        clip_status_path: Path,
        manifest_path: Path,
        blind: bool = False,
    ):
        self.labels_path = Path(labels_path)
        self.clip_status_path = Path(clip_status_path)
        self.manifest_path = Path(manifest_path)
        self.blind = blind

        self.labels_df: pd.DataFrame = pd.DataFrame(columns=LABELS_COLUMNS)
        self.status_df: pd.DataFrame = pd.DataFrame(columns=CLIP_STATUS_COLUMNS)
        self.manifest_df: pd.DataFrame = pd.DataFrame()

        self.load()

    def load(self) -> None:
        """Load manifest, clip status, and labels from disk."""
        if self.manifest_path.exists():
            self.manifest_df = pd.read_csv(self.manifest_path, dtype=str)
        else:
            self.manifest_df = pd.DataFrame()

        if self.clip_status_path.exists():
            df = pd.read_csv(self.clip_status_path, dtype=str)
            validate_clip_status_df(df)
            self.status_df = df
        else:
            self.status_df = pd.DataFrame(columns=CLIP_STATUS_COLUMNS)

        if self.labels_path.exists():
            df = pd.read_csv(self.labels_path, dtype=str)
            validate_labels_df(df)
            self.labels_df = df
        else:
            self.labels_df = pd.DataFrame(columns=LABELS_COLUMNS)

    def save_labels(self) -> None:
        """Persist labels dataframe to CSV."""
        self.labels_path.parent.mkdir(parents=True, exist_ok=True)
        # Ensure correct column order
        out_df = self.labels_df[LABELS_COLUMNS].copy()
        out_df.to_csv(self.labels_path, index=False)

    def save_status(self) -> None:
        """Persist clip status dataframe to CSV."""
        self.clip_status_path.parent.mkdir(parents=True, exist_ok=True)
        out_df = self.status_df[CLIP_STATUS_COLUMNS].copy()
        out_df.to_csv(self.clip_status_path, index=False)

    def save_manifest(self) -> None:
        """Persist manifest dataframe to CSV."""
        if not self.manifest_df.empty:
            self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
            self.manifest_df.to_csv(self.manifest_path, index=False)

    def get_fps_for_clip(self, clip_id: str) -> float:
        """Retrieve the verified FPS for a given clip from manifest."""
        if self.manifest_df.empty:
            return 60.0
        row = self.manifest_df[self.manifest_df["clip_id"] == clip_id]
        if not row.empty:
            try:
                return float(row.iloc[0]["fps"])
            except (ValueError, TypeError):
                pass
        return 60.0

    def get_view_for_clip(self, clip_id: str) -> str:
        """Get the current view assignment for a clip."""
        if self.manifest_df.empty:
            return ""
        row = self.manifest_df[self.manifest_df["clip_id"] == clip_id]
        if not row.empty:
            v = row.iloc[0].get("view", "")
            return "" if pd.isna(v) else str(v).strip()
        return ""

    def set_clip_view(self, clip_id: str, view: str) -> None:
        """Set view in manifest and save immediately."""
        if view not in VALID_VIEWS:
            raise ValueError(f"Invalid view '{view}'. Must be one of {VALID_VIEWS}")
        if not self.manifest_df.empty:
            mask = self.manifest_df["clip_id"] == clip_id
            if mask.any():
                self.manifest_df.loc[mask, "view"] = view
                self.save_manifest()

    def get_marks_for_clip(self, clip_id: str) -> List[Dict[str, Any]]:
        """Return all marks recorded for a clip (hidden if in blind mode)."""
        if self.blind:
            return []
        rows = self.labels_df[self.labels_df["clip_id"] == clip_id]
        return rows.to_dict(orient="records")

    def is_clip_reviewed(self, clip_id: str) -> bool:
        """Check if clip has been explicitly reviewed."""
        row = self.status_df[self.status_df["clip_id"] == clip_id]
        if not row.empty:
            return str(row.iloc[0]["reviewed"]).lower() == "yes"
        return False

    def add_mark(
        self,
        clip_id: str,
        frame_index: int,
        kind: str = "delivery",
        notes: str = "",
    ) -> Dict[str, Any]:
        """
        Record a delivery or partial delivery at exact frame_index.
        frame_index is the stored truth; delivery_time_s = frame_index / fps.
        """
        if kind not in VALID_KINDS:
            raise ValueError(f"Invalid kind '{kind}'. Must be one of {VALID_KINDS}")

        fps = self.get_fps_for_clip(clip_id)
        time_s = f"{(frame_index / fps):.4f}"

        # If an identical mark already exists on this frame, return it without duplicate
        existing_dup = self.labels_df[
            (self.labels_df["clip_id"] == clip_id)
            & (self.labels_df["frame_index"] == str(frame_index))
            & (self.labels_df["kind"] == kind)
        ]
        if not existing_dup.empty:
            return existing_dup.iloc[0].to_dict()

        # If clip had a kind='none' row, remove it since a real delivery is being added
        self.labels_df = self.labels_df[
            ~((self.labels_df["clip_id"] == clip_id) & (self.labels_df["kind"] == "none"))
        ]

        new_row = {
            "clip_id": clip_id,
            "delivery_time_s": str(time_s),
            "frame_index": str(frame_index),
            "kind": kind,
            "notes": notes.strip(),
            "schema_version": SCHEMA_VERSION,
        }
        self.labels_df = pd.concat([self.labels_df, pd.DataFrame([new_row])], ignore_index=True)
        self.save_labels()
        return new_row

    def undo_last_mark(self, clip_id: str) -> Optional[Dict[str, Any]]:
        """Remove the most recently added mark for this clip."""
        clip_marks = self.labels_df[self.labels_df["clip_id"] == clip_id]
        if clip_marks.empty:
            return None
        last_index = clip_marks.index[-1]
        removed_row = self.labels_df.loc[last_index].to_dict()
        self.labels_df = self.labels_df.drop(last_index).reset_index(drop=True)
        self.save_labels()
        return removed_row

    def set_clip_reviewed(self, clip_id: str, reviewed: bool = True) -> None:
        """
        Mark clip as reviewed.
        If clip has no delivery marks when reviewed, write exactly one row with kind='none'.
        """
        now_str = datetime.now(timezone.utc).isoformat() if reviewed else ""
        rev_str = "yes" if reviewed else "no"

        # Update status_df
        mask = self.status_df["clip_id"] == clip_id
        if mask.any():
            self.status_df.loc[mask, "reviewed"] = rev_str
            self.status_df.loc[mask, "reviewed_at"] = now_str
        else:
            new_row = {
                "clip_id": clip_id,
                "reviewed": rev_str,
                "reviewed_at": now_str,
            }
            self.status_df = pd.concat([self.status_df, pd.DataFrame([new_row])], ignore_index=True)
        self.save_status()

        if reviewed:
            # Check existing marks for this clip
            clip_marks = self.labels_df[self.labels_df["clip_id"] == clip_id]
            active_marks = clip_marks[clip_marks["kind"].isin(["delivery", "partial_start", "partial_end"])]
            if active_marks.empty:
                # Write one row with empty delivery_time_s and frame_index, kind='none'
                none_row = {
                    "clip_id": clip_id,
                    "delivery_time_s": "",
                    "frame_index": "",
                    "kind": "none",
                    "notes": "",
                    "schema_version": SCHEMA_VERSION,
                }
                # Remove any duplicate none rows if present
                self.labels_df = self.labels_df[
                    ~((self.labels_df["clip_id"] == clip_id) & (self.labels_df["kind"] == "none"))
                ]
                self.labels_df = pd.concat([self.labels_df, pd.DataFrame([none_row])], ignore_index=True)
                self.save_labels()
            else:
                # Remove any stray 'none' row if active marks exist
                self.labels_df = self.labels_df[
                    ~((self.labels_df["clip_id"] == clip_id) & (self.labels_df["kind"] == "none"))
                ]
                self.save_labels()
