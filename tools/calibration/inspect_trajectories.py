import json

with open("outputs/trajectories/dev_trajectories.json") as f:
    trajs = json.load(f)

for t in trajs:
    pts = t["points"]
    d_idx = t["delivery_idx"]
    clip = t["clip_id"]
    cand = t["candidate_time_s"]
    print(f"D{d_idx:02d} [{clip} @ {cand:.2f}s]: {len(pts)} pts | rel=({pts[0]['x']:.1f}, {pts[0]['y']:.1f}) f{pts[0]['frame_idx']} | bnc=({t['bounce_x']:.1f}, {t['bounce_y']:.1f}) | imp=({pts[-1]['x']:.1f}, {pts[-1]['y']:.1f}) f{pts[-1]['frame_idx']}")

