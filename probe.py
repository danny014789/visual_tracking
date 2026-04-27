import pyrealsense2 as rs

ctx = rs.context()
devices = list(ctx.query_devices())
print(f"Found {len(devices)} device(s)")
for d in devices:
    print(f"  - {d.get_info(rs.camera_info.name)}  SN={d.get_info(rs.camera_info.serial_number)}")
    for s in d.query_sensors():
        sname = s.get_info(rs.camera_info.name)
        print(f"    Sensor: {sname}")
        seen = set()
        for p in s.get_stream_profiles():
            vp = p.as_video_stream_profile()
            key = (str(p.stream_type()), vp.width(), vp.height(), str(p.format()), vp.fps())
            if key in seen:
                continue
            seen.add(key)
            print(f"      {key[0]:>20s}  {vp.width()}x{vp.height()}  {key[3]:>12s}  {vp.fps()}fps")
