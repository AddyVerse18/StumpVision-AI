import json

with open("outputs/trajectories/dev_trajectories.json") as f:
    trajs = json.load(f)

for idx, t in enumerate(trajs, 1):
    pts = t.get("points", [])
    f_pt = pts[0] if pts else {}
    l_pt = pts[-1] if pts else {}
    bx, by = t.get("bounce_x"), t.get("bounce_y")
    fx, fy = f_pt.get("x", 0), f_pt.get("y", 0)
    lx, ly = l_pt.get("x", 0), l_pt.get("y", 0)
    sld = t.get("stump_line_display")
    kmh = t.get("estimated_speed_kmh")
    print(f"D{idx:02d}: pts={len(pts)}, rel=({fx:.1f}, {fy:.1f}), bnc=({bx}, {by}), imp=({lx:.1f}, {ly:.1f}), line={sld}, kmh={kmh}")

