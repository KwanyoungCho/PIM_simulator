# 버스 예약 순서 시뮬레이션
events = [
    ("533.03", "c2f_3_final compute done", "schedule write"),
    ("533.04", "tile3 ready", "schedule read"),
    ("533.04", "tile4 ready", "schedule read"),
    ("536.04", "tile2 compute done (537.54)", "schedule write"),
    ("537.54", "tile2 compute done", "actual schedule write"),
    ("537.55", "reduce compute done", "schedule write"),
]

bus_busy = 530.02
print(f"Initial bus_busy: {bus_busy}\n")

# 1. c2f_3_final write 스케줄 (533.03 compute done)
write_start = max(533.04, bus_busy + 0.000001)
write_end = write_start + 125.00
bus_busy = write_end
print(f"1. c2f_3_final write: {write_start:.2f}~{write_end:.2f}, bus_busy={bus_busy:.2f}")

# 2. tile3 read 스케줄 (533.04 ready)
read_start = max(533.04, bus_busy + 0.000001)
read_end = read_start + 125.00
bus_busy = read_end
print(f"2. tile3 read: {read_start:.2f}~{read_end:.2f}, bus_busy={bus_busy:.2f}")

# 3. tile4 read 스케줄 (533.04 ready)
read_start = max(533.04, bus_busy + 0.000001)
read_end = read_start + 125.00
bus_busy = read_end
print(f"3. tile4 read: {read_start:.2f}~{read_end:.2f}, bus_busy={bus_busy:.2f}")

# 4. tile2 write 스케줄 (537.54 compute done)
write_start = max(537.54, bus_busy + 0.000001)
write_end = write_start + 15.62
bus_busy = write_end
print(f"4. tile2 write: {write_start:.2f}~{write_end:.2f}, bus_busy={bus_busy:.2f}")

# 5. reduce write 스케줄 (537.55 compute done)
write_start = max(537.55, bus_busy + 0.000001)
write_end = write_start + 15.62
bus_busy = write_end
print(f"5. reduce write: {write_start:.2f}~{write_end:.2f}, bus_busy={bus_busy:.2f}")

print(f"\n예상: tile2 write should start at {908.04:.2f}")
print(f"실제: tile2 write starts at 1283.04")
print(f"차이: {1283.04 - 908.04:.2f} us")
