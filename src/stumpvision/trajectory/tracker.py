"""
tracker.py
==========
M5 Ball Pitch Trajectory Tracker & Delivery Windowing for StumpVision AI.

Tracks cricket ball delivery trajectories from release through pitch bounce
to the striker batsman crease. Computes pitching length, bounce coordinates,
delivery event start/end windows, and velocity profiles.
"""

from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import cv2
import numpy as np
import pandas as pd


PITCH_LENGTH_METERS = 20.12  # Standard cricket pitch length (stump to stump)


@dataclass
class TrajectoryPoint:
    frame_idx: int
    time_s: float
    x: float
    y: float
    radius: float
    area: float
    confidence: float = 1.0


@dataclass
class DeliveryTrajectory:
    clip_id: str
    delivery_idx: int
    candidate_time_s: float
    points: List[TrajectoryPoint] = field(default_factory=list)
    t_release: Optional[float] = None
    t_bounce: Optional[float] = None
    t_impact: Optional[float] = None
    bounce_x: Optional[float] = None
    bounce_y: Optional[float] = None
    length_classification: str = "unpitched"  # 'yorker', 'full', 'good_length', 'short', 'unpitched'
    line_classification: str = "middle"      # 'off', 'middle', 'leg'
    stump_line: str = "middle_stump"         # granular: 'wide_off', 'outside_off', 'off_stump', 'middle_stump', 'leg_stump', 'outside_leg', 'wide_leg'
    stump_line_display: str = "Middle Stump" # human-readable: 'Wide Outside Off', 'Outside Off', 'Off Stump', 'Middle Stump', 'Leg Stump', etc.
    mean_speed_px_per_s: float = 0.0
    estimated_speed_kmh: float = 0.0
    speed_release_kmh: float = 0.0
    speed_release_mph: float = 0.0
    drs_decision: str = "NOT OUT"
    drs_wickets: str = "Missing"
    is_valid: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["points"] = [asdict(p) for p in self.points]
        return d


class BallTrajectoryTracker:
    """
    Cricket ball delivery trajectory extractor and pitch bounce estimator.
    """

    def __init__(
        self,
        corridor_x_min: int = 500,
        corridor_x_max: int = 960,
        corridor_y_min: int = 50,
        corridor_y_max: int = 580,
        diff_threshold: int = 20,
        brightness_threshold: int = 170,
        min_blob_area: float = 4.0,
        max_blob_area: float = 180.0,
        min_track_points: int = 5,
        fps: float = 60.0,
        poses_path: Optional[Union[str, Path]] = None,
    ):
        self.corridor_x_min = corridor_x_min
        self.corridor_x_max = corridor_x_max
        self.corridor_y_min = corridor_y_min
        self.corridor_y_max = corridor_y_max
        self.diff_threshold = diff_threshold
        self.brightness_threshold = brightness_threshold
        self.min_blob_area = min_blob_area
        self.max_blob_area = max_blob_area
        self.min_track_points = min_track_points
        self.fps = fps

        # Load batsman pose keypoints to anchor exact bat contact target
        self.poses_data: List[dict] = []
        p_path = Path(poses_path) if poses_path else Path("labels/m4_poses/batsman_pose_dev.json")
        if p_path.exists():
            try:
                with open(p_path, "r", encoding="utf-8") as f:
                    self.poses_data = json.load(f)
            except Exception:
                self.poses_data = []

    def find_bat_contact(
        self,
        clip_id: str,
        candidate_time_s: float,
        bounce_f: int,
        length_cat: str,
    ) -> Optional[Tuple[int, float, float]]:
        """
        Locate the exact bat contact coordinates (frame_idx, bat_x, bat_y)
        from striker batsman biomechanical pose keypoints at shot execution.
        """
        if not self.poses_data:
            return None

        c_poses = [p for p in self.poses_data if p.get("file_name", "").startswith(clip_id)]
        if not c_poses:
            return None

        def _p_time(p: dict) -> float:
            fn = p.get("file_name", "")
            fidx_str = fn.split("_")[-1].split(".")[0].replace("f", "")
            try:
                return int(fidx_str) / float(self.fps)
            except Exception:
                return 0.0

        # Find 3 closest poses around candidate delivery
        c_poses_sorted = sorted(c_poses, key=lambda p: abs(_p_time(p) - candidate_time_s))
        cand_poses = sorted(
            c_poses_sorted[:3],
            key=lambda p: int(p.get("file_name", "").split("_")[-1].split(".")[0].replace("f", "") or 0),
        )
        if not cand_poses:
            return None

        # Latest pose in delivery window is shot contact
        shot_pose = cand_poses[-1]
        try:
            fn = shot_pose.get("file_name", "")
            shot_f = int(fn.split("_")[-1].split(".")[0].replace("f", ""))
        except Exception:
            shot_f = bounce_f + 6

        kp = shot_pose.get("keypoints", [])
        if len(kp) >= 17:
            # Wrists: 9 (left), 10 (right)
            w_x = (kp[9][0] + kp[10][0]) / 2.0
            w_y = (kp[9][1] + kp[10][1]) / 2.0
            # Ankles: 15 (left), 16 (right)
            ank_y = (kp[15][1] + kp[16][1]) / 2.0
            bat_x = float(w_x)
            # Bat face sweet spot lies between hands and popping crease turf
            bat_y = float(w_y + 0.45 * (ank_y - w_y))
        else:
            bat_x = 895.0
            bat_y = 535.0

        # Realistic post-bounce flight duration to bat
        if length_cat == "short":
            eff_shot_f = max(shot_f, bounce_f + 12)
        elif length_cat == "good_length":
            eff_shot_f = max(shot_f, bounce_f + 6)
        else:
            # Yorker / Full
            eff_shot_f = max(shot_f, bounce_f + 2)

        return eff_shot_f, bat_x, bat_y

    def extract_blobs_from_frames(
        self,
        frames: List[np.ndarray],
        start_frame_idx: int,
    ) -> List[List[TrajectoryPoint]]:
        """Extract candidate ball blobs from sequential video frames."""
        blobs_per_frame: List[List[TrajectoryPoint]] = []

        for i in range(1, len(frames)):
            f_idx = start_frame_idx + i
            t_s = round(f_idx / self.fps, 4) if self.fps > 0 else 0.0
            f_prev = frames[i - 1]
            f_curr = frames[i]

            gray_curr = cv2.cvtColor(f_curr, cv2.COLOR_BGR2GRAY)
            gray_prev = cv2.cvtColor(f_prev, cv2.COLOR_BGR2GRAY)
            diff1 = cv2.absdiff(gray_curr, gray_prev)

            # Also check 2-frame step to defeat 30/60 FPS interlaced duplicate frames
            if i >= 2:
                gray_step2 = cv2.cvtColor(frames[i - 2], cv2.COLOR_BGR2GRAY)
                diff2 = cv2.absdiff(gray_curr, gray_step2)
                diff = cv2.max(diff1, diff2)
            else:
                diff = diff1

            # Brightness + difference mask
            mask = (diff > self.diff_threshold) & (gray_curr > self.brightness_threshold)

            # Restrict to pitch delivery corridor
            corridor = np.zeros_like(mask)
            corridor[
                self.corridor_y_min : self.corridor_y_max,
                self.corridor_x_min : self.corridor_x_max,
            ] = True

            active = (mask & corridor).astype(np.uint8) * 255
            contours, _ = cv2.findContours(active, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            frame_blobs: List[TrajectoryPoint] = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if self.min_blob_area <= area <= self.max_blob_area:
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect = float(w) / max(1, h)
                    if 0.35 <= aspect <= 2.5:
                        cx = float(x + w / 2.0)
                        cy = float(y + h / 2.0)
                        radius = float(max(w, h) / 2.0)
                        conf = min(1.0, float(np.mean(gray_curr[y : y + h, x : x + w])) / 255.0)
                        frame_blobs.append(
                            TrajectoryPoint(
                                frame_idx=f_idx,
                                time_s=t_s,
                                x=cx,
                                y=cy,
                                radius=radius,
                                area=area,
                                confidence=conf,
                            )
                        )
            blobs_per_frame.append(frame_blobs)

        return blobs_per_frame

    def link_trajectory_points(
        self,
        blobs_per_frame: List[List[TrajectoryPoint]],
        start_frame_idx: int,
    ) -> List[TrajectoryPoint]:
        """
        Greedy trajectory linker with forward velocity projection and physical constraints.
        """
        all_tracks: List[List[TrajectoryPoint]] = []

        for frame_i, blobs in enumerate(blobs_per_frame):
            for b in blobs:
                track = [b]
                cur_b = b
                for next_i in range(frame_i + 1, min(len(blobs_per_frame), frame_i + 15)):
                    next_blobs = blobs_per_frame[next_i]
                    dt = next_i - (cur_b.frame_idx - start_frame_idx - 1)
                    if dt <= 0:
                        continue

                    best_match = None
                    best_err = 999.0

                    for nb in next_blobs:
                        dx = nb.x - cur_b.x
                        dy = nb.y - cur_b.y
                        dist = float(np.hypot(dx, dy))

                        # Cricket ball down-pitch: dx mostly positive (toward batsman), dy downwards
                        if 3.0 <= dist <= 80.0 * dt and dx >= -8.0:
                            err = abs(dist - 25.0 * dt)
                            if err < best_err:
                                best_err = err
                                best_match = nb

                    if best_match is not None and best_err < 45.0:
                        track.append(best_match)
                        cur_b = best_match

                if len(track) >= self.min_track_points:
                    all_tracks.append(track)

        if not all_tracks:
            return []

        # Return longest, most continuous track
        return max(all_tracks, key=lambda tr: len(tr))

    def analyze_trajectory(
        self,
        clip_id: str,
        delivery_idx: int,
        candidate_time_s: float,
        points: List[TrajectoryPoint],
        bat_contact: Optional[Tuple[int, float, float]] = None,
    ) -> DeliveryTrajectory:
        """
        Analyze physical properties, bounce point, and length classification of trajectory.
        Extends trajectory from turf bounce to batsman bat contact.
        """
        if len(points) < self.min_track_points:
            return DeliveryTrajectory(
                clip_id=clip_id,
                delivery_idx=delivery_idx,
                candidate_time_s=candidate_time_s,
                points=points,
                is_valid=False,
            )

        t_release = points[0].time_s

        # 1. Detect pitch turf bounce / inflection point from pre-bounce flight
        y_vals = [p.y for p in points]
        max_y_idx = int(np.argmax(y_vals))

        bounce_pt = points[max_y_idx]
        bounce_x = bounce_pt.x
        bounce_y = bounce_pt.y
        t_bounce = bounce_pt.time_s

        # 2. Pitching Length Classification (Camera 1 Perspective):
        # Far stumps at y ~ 500-520
        # Full/Yorker: y >= 470
        # Good length: 420 <= y < 470
        # Short pitch: y < 420
        if bounce_y >= 485.0:
            length_cat = "yorker"
        elif bounce_y >= 460.0:
            length_cat = "full"
        elif bounce_y >= 415.0:
            length_cat = "good_length"
        else:
            length_cat = "short"

        # 3. Extend trajectory points with physically constrained post-bounce momentum
        descent_pts = points[: max_y_idx + 1]
        all_pts = list(descent_pts)

        if bat_contact is None:
            bat_contact = self.find_bat_contact(
                clip_id=clip_id,
                candidate_time_s=candidate_time_s,
                bounce_f=bounce_pt.frame_idx,
                length_cat=length_cat,
            )

        # Calculate incident horizontal drift rate from pre-bounce flight
        if len(descent_pts) >= 2:
            prev_pt = descent_pts[-3] if len(descent_pts) >= 3 else descent_pts[-2]
            delta_y = max(1.0, bounce_pt.y - prev_pt.y)
            drift_rate = (bounce_pt.x - prev_pt.x) / delta_y
        else:
            drift_rate = 0.15

        if bat_contact is not None:
            shot_f, bat_target_x, bat_target_y = bat_contact
            f_bnc = bounce_pt.frame_idx

            target_y = min(545.0, max(bounce_pt.y + 15.0, bat_target_y))
            natural_x = bounce_pt.x + drift_rate * (target_y - bounce_pt.y)

            # Enforce physical forward momentum: ball cannot reverse lateral direction mid-air
            if drift_rate > 0 and bat_target_x < bounce_pt.x:
                bat_target_x = natural_x
            elif drift_rate < 0 and bat_target_x > bounce_pt.x:
                bat_target_x = natural_x

            # If batsman stance is in line with the ball delivery corridor (< 35px), ball meets bat face
            if abs(natural_x - bat_target_x) < 35.0:
                final_x = bat_target_x
            else:
                final_x = natural_x

            if shot_f > f_bnc:
                step_count = max(1, shot_f - f_bnc)
                frame_stride = 2 if step_count >= 6 else 1
                for f in range(f_bnc + frame_stride, shot_f + 1, frame_stride):
                    alpha = float(f - f_bnc) / float(step_count)
                    px = (1.0 - alpha) * bounce_x + alpha * final_x
                    py = (1.0 - alpha) * bounce_y + alpha * target_y
                    all_pts.append(
                        TrajectoryPoint(
                            frame_idx=f,
                            time_s=round(f / float(self.fps), 4),
                            x=round(float(px), 1),
                            y=round(float(py), 1),
                            radius=bounce_pt.radius,
                            area=bounce_pt.area,
                            confidence=round(bounce_pt.confidence * 0.95, 2),
                        )
                    )
                if all_pts[-1].frame_idx != shot_f:
                    all_pts.append(
                        TrajectoryPoint(
                            frame_idx=shot_f,
                            time_s=round(shot_f / float(self.fps), 4),
                            x=round(float(final_x), 1),
                            y=round(float(target_y), 1),
                            radius=bounce_pt.radius,
                            area=bounce_pt.area,
                            confidence=round(bounce_pt.confidence * 0.95, 2),
                        )
                    )

        t_impact = all_pts[-1].time_s

        # 4. Pitching Line & Batter's End Stump Line Classification
        from stumpvision.trajectory.stumps import classify_stump_line

        # Estimate where the ball trajectory intersects the batter's crease (y = 540)
        last_pt = all_pts[-1]
        if last_pt.y >= 530.0:
            x_crease = last_pt.x
        elif len(all_pts) >= 2:
            prev_pt = all_pts[-2]
            dy_step = max(1.0, last_pt.y - prev_pt.y)
            dx_step = last_pt.x - prev_pt.x
            x_crease = last_pt.x + (dx_step / dy_step) * (540.0 - last_pt.y)
        else:
            x_crease = bounce_x

        stump_line_id, stump_line_display, line_cat = classify_stump_line(x_crease)

        # 5. Compute telemetry metrics
        dur_s = max(0.01, t_impact - t_release)
        total_dist_px = sum(
            np.hypot(all_pts[i].x - all_pts[i - 1].x, all_pts[i].y - all_pts[i - 1].y)
            for i in range(1, len(all_pts))
        )
        mean_speed_px = float(total_dist_px / dur_s)
        flight_duration_s = dur_s
        est_speed_kmh = float((PITCH_LENGTH_METERS / max(0.30, flight_duration_s)) * 3.6)

        return DeliveryTrajectory(
            clip_id=clip_id,
            delivery_idx=delivery_idx,
            candidate_time_s=candidate_time_s,
            points=all_pts,
            t_release=t_release,
            t_bounce=t_bounce,
            t_impact=t_impact,
            bounce_x=bounce_x,
            bounce_y=bounce_y,
            length_classification=length_cat,
            line_classification=line_cat,
            stump_line=stump_line_id,
            stump_line_display=stump_line_display,
            mean_speed_px_per_s=round(mean_speed_px, 1),
            estimated_speed_kmh=round(est_speed_kmh, 1),
            is_valid=True,
        )

    def track_clip_delivery(
        self,
        video_path: Union[str, Path],
        clip_id: str,
        delivery_idx: int,
        candidate_time_s: float,
        search_window_s: float = 0.5,
    ) -> DeliveryTrajectory:
        """
        Track trajectory for a single confirmed delivery in video file.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            return DeliveryTrajectory(
                clip_id=clip_id,
                delivery_idx=delivery_idx,
                candidate_time_s=candidate_time_s,
                is_valid=False,
            )

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return DeliveryTrajectory(
                clip_id=clip_id,
                delivery_idx=delivery_idx,
                candidate_time_s=candidate_time_s,
                is_valid=False,
            )

        center_frame = int(round(candidate_time_s * self.fps))
        start_frame = max(0, center_frame - int(search_window_s * self.fps))
        end_frame = center_frame + int(search_window_s * self.fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        for _ in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if len(frames) < 2:
            return DeliveryTrajectory(
                clip_id=clip_id,
                delivery_idx=delivery_idx,
                candidate_time_s=candidate_time_s,
                is_valid=False,
            )

        blobs = self.extract_blobs_from_frames(frames, start_frame)
        track_points = self.link_trajectory_points(blobs, start_frame)
        return self.analyze_trajectory(clip_id, delivery_idx, candidate_time_s, track_points)


def track_all_deliveries(
    candidates_csv_path: Union[str, Path],
    manifest_csv_path: Union[str, Path],
    data_dir: Union[str, Path],
    output_json_path: Optional[Union[str, Path]] = None,
    output_summary_csv_path: Optional[Union[str, Path]] = None,
    search_window_s: float = 0.5,
) -> Tuple[List[DeliveryTrajectory], pd.DataFrame]:
    """
    Track pitch trajectories for all confirmed delivery candidates across video clips.
    """
    cand_df = pd.read_csv(candidates_csv_path)
    manifest_df = pd.read_csv(manifest_csv_path)
    data_dir = Path(data_dir)

    manifest_map = dict(zip(manifest_df["clip_id"], manifest_df["relative_path"]))
    tracker = BallTrajectoryTracker()

    trajectories: List[DeliveryTrajectory] = []
    summary_rows: List[Dict[str, Any]] = []

    for idx, row in cand_df.iterrows():
        cid = str(row["clip_id"]).strip()
        t_cand = float(row["time_s"])
        rel_path = manifest_map.get(cid, f"raw/cam1/{cid}.mp4")
        video_full = data_dir / rel_path

        traj = tracker.track_clip_delivery(
            video_path=video_full,
            clip_id=cid,
            delivery_idx=idx + 1,
            candidate_time_s=t_cand,
            search_window_s=search_window_s,
        )
        trajectories.append(traj)

        flight_dur = (
            round(traj.t_impact - traj.t_release, 3)
            if (traj.t_impact is not None and traj.t_release is not None)
            else None
        )

        from stumpvision.trajectory.drs import evaluate_drs
        drs = evaluate_drs(traj)

        summary_rows.append({
            "clip_id": cid,
            "delivery_idx": idx + 1,
            "candidate_time_s": t_cand,
            "is_valid": traj.is_valid,
            "point_count": len(traj.points),
            "flight_duration_s": flight_dur,
            "t_release": traj.t_release,
            "t_bounce": traj.t_bounce,
            "t_impact": traj.t_impact,
            "bounce_x": round(traj.bounce_x, 1) if traj.bounce_x else None,
            "bounce_y": round(traj.bounce_y, 1) if traj.bounce_y else None,
            "length": traj.length_classification,
            "line": traj.line_classification,
            "stump_line": getattr(traj, "stump_line", "middle_stump"),
            "stump_line_display": getattr(traj, "stump_line_display", "Middle Stump"),
            "speed_kmh": drs.speed_release_kmh,
            "speed_mph": drs.speed_release_mph,
            "speed_post_kmh": drs.speed_post_bounce_kmh,
            "pace_category": drs.pace_category,
            "drs_pitching": drs.pitching_display,
            "drs_impact": drs.impact_display,
            "drs_wickets": drs.wickets_display,
            "drs_decision": drs.decision,
            "drs_reason": drs.decision_reason,
            "arrival_height_cm": drs.arrival_height_cm,
        })

    summary_df = pd.DataFrame(summary_rows)

    if output_json_path:
        out_j = Path(output_json_path)
        out_j.parent.mkdir(parents=True, exist_ok=True)
        with open(out_j, "w", encoding="utf-8") as f:
            json.dump([t.to_dict() for t in trajectories], f, indent=2)

    if output_summary_csv_path:
        out_c = Path(output_summary_csv_path)
        out_c.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(out_c, index=False)

    return trajectories, summary_df

