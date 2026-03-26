"""
Thu thập dữ liệu từ DWM1001 tag qua serial → lưu CSV tương thích ECA-UWB.

Firmware output (110 cột — CH/FRAME_LEN/PREAM_LEN/BITRATE/PRFR đã bỏ ở firmware):
  Col 0       : NLOS label  (0=LOS, 1=NLOS — set g_label trong firmware)
  Col 1       : RANGE       (mm)
  Col 2       : FP_IDX
  Col 3–5     : FP_AMP1, FP_AMP2, FP_AMP3
  Col 6       : STDEV_NOISE
  Col 7       : CIR_PWR
  Col 8       : MAX_NOISE
  Col 9       : RXPACC
  Col 10–109  : CIR0 … CIR99

CSV xuất ra giống hệt firmware (110 cột, tương thích config.py VNU_CIR_START=10).

Cách dùng:
    python collect_vnu.py --env lab     --label los  --n 1500
    python collect_vnu.py --env lab     --label nlos --n 1500
    python collect_vnu.py --env hallway --label los  --n 1500
    python collect_vnu.py --monitor
"""

import argparse
import sys
import time
from pathlib import Path

import serial

# ── Cấu hình port / baud ───────────────────────────────────────────────────────
PORT = "/dev/ttyACM0"
BAUD = 912600

# ── Thư mục lưu ───────────────────────────────────────────────────────────────
SAVE_DIR = Path.home() / "Documents/UWB-NLOS/VNU_dataset"

# ── Layout firmware output ────────────────────────────────────────────────────
# Firmware output 110 cột (CH/FRAME_LEN/PREAM_LEN/BITRATE/PRFR đã bỏ):
#   Col 0       : NLOS label (0/1)
#   Col 1       : RANGE (mm)
#   Col 2       : FP_IDX
#   Col 3–5     : FP_AMP1, FP_AMP2, FP_AMP3
#   Col 6       : STDEV_NOISE
#   Col 7       : CIR_PWR
#   Col 8       : MAX_NOISE
#   Col 9       : RXPACC
#   Col 10–109  : CIR0 … CIR99
FW_TOTAL_COLS = 110
FW_CIR_START  = 10   # CIR0 bắt đầu ở cột 10

# ── Sanity-check ranges (phát hiện byte-drop còn sót) ─────────────────────────
RANGE_MIN_MM  = 50      # khoảng cách tối thiểu (mm)
RANGE_MAX_MM  = 20000   # khoảng cách tối đa (mm)
FP_IDX_MIN    = 500
FP_IDX_MAX    = 1016
RXPACC_MIN    = 50
RXPACC_MAX    = 200


# ── Header CSV xuất ra ────────────────────────────────────────────────────────
CIR_HEADER   = ",".join([f"CIR{i}" for i in range(100)])
OUTPUT_HEADER = (
    f"NLOS,RANGE,FP_IDX,FP_AMP1,FP_AMP2,FP_AMP3,"
    f"STDEV_NOISE,CIR_PWR,MAX_NOISE,RXPACC,{CIR_HEADER}"
)


def parse_fw_line(line: str, expected_label: str):
    """
    Parse một dòng CSV từ firmware (110 cột).
    Trả về chuỗi CSV output (110 cột) hoặc None nếu dòng không hợp lệ.

    Layout firmware = layout CSV output (không cần bỏ cột nào):
      Col 0: NLOS, Col 1-9: RANGE…RXPACC, Col 10-109: CIR0…CIR99
    """
    cols = line.split(",")
    if len(cols) < FW_TOTAL_COLS:
        return None

    try:
        label  = cols[0].strip()
        rng    = int(cols[1])
        fp_idx = int(cols[2])
        rxpacc = int(cols[9])
    except ValueError:
        return None

    # Kiểm tra label khớp với --label arg
    if label != expected_label:
        return None

    # Sanity check để lọc dòng bị byte-drop
    if not (RANGE_MIN_MM <= rng <= RANGE_MAX_MM):
        return None
    if not (FP_IDX_MIN <= fp_idx <= FP_IDX_MAX):
        return None
    if not (RXPACC_MIN <= rxpacc <= RXPACC_MAX):
        return None

    # Output: toàn bộ 110 cột giữ nguyên (label + diag + CIR)
    diag_part = ",".join(cols[1:10])                          # RANGE…RXPACC
    cir_part  = ",".join(cols[FW_CIR_START:FW_CIR_START + 100])
    return f"{label},{diag_part},{cir_part}"


def collect(env: str, label: str, n_samples: int, port: str):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SAVE_DIR / f"{env}_{label}.csv"

    expected_label = "0" if label == "los" else "1"

    print(f"Port      : {port}  @{BAUD}")
    print(f"Môi trường: {env}")
    print(f"Label     : {label} ({expected_label})")
    print(f"Mục tiêu  : {n_samples} mẫu")
    print(f"Lưu vào   : {out_path}")
    print()
    input("Nhấn Enter để bắt đầu thu thập, Ctrl+C để hủy...\n")

    ser = serial.Serial(port, BAUD, timeout=2)
    time.sleep(0.5)
    ser.reset_input_buffer()

    count   = 0
    skipped = 0

    with open(out_path, "w") as f:
        f.write(OUTPUT_HEADER + "\n")
        print("Đang thu thập... (Ctrl+C để dừng sớm)\n")
        try:
            while count < n_samples:
                raw = ser.readline()
                if not raw:
                    continue

                line = raw.decode("ascii", errors="ignore").strip()
                if not line:
                    continue

                # Bỏ qua dòng header/debug từ firmware (không bắt đầu bằng chữ số)
                if not line[0].isdigit():
                    continue

                row = parse_fw_line(line, expected_label)
                if row is None:
                    skipped += 1
                    continue

                f.write(row + "\n")
                f.flush()
                count += 1

                if count % 50 == 0:
                    pct = count / n_samples * 100
                    bar = "#" * (count * 20 // n_samples)
                    print(f"\r  [{bar:<20}] {count}/{n_samples} ({pct:.0f}%)  skip={skipped}",
                          end="", flush=True)

        except KeyboardInterrupt:
            print(f"\n\nDừng sớm theo yêu cầu.")

    ser.close()
    print(f"\n\nHoàn thành: {count} mẫu → {out_path}")
    if skipped:
        print(f"Bỏ qua     : {skipped} dòng lỗi/sanity-fail")


def monitor(port: str):
    """In thô mọi thứ nhận được từ serial — dùng để debug firmware."""
    print(f"Monitor {port} @{BAUD}  (Ctrl+C để thoát)\n")
    try:
        ser = serial.Serial(port, BAUD, timeout=2)
        ser.reset_input_buffer()
        while True:
            raw = ser.readline()
            if raw:
                line = raw.decode("ascii", errors="replace").rstrip()
                cols = line.split(",")
                # Hiển thị số cột để dễ kiểm tra format
                print(f"[{len(cols):3d} cols]  {line[:120]}")
            else:
                print("[timeout]")
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()


def main():
    parser = argparse.ArgumentParser(description="Thu thập dữ liệu VNU UWB")
    parser.add_argument("--env",   default=None,
                        choices=["room", "hallway", "garage"],
                        help="Môi trường đo (room=trong phòng, hallway=hành lang, garage=nhà để xe)")
    parser.add_argument("--label", default=None,
                        choices=["los", "nlos"],
                        help="LOS hay NLOS")
    parser.add_argument("--n",     type=int, default=1500,
                        help="Số mẫu cần thu (default: 1500)")
    parser.add_argument("--port",  default=PORT,
                        help=f"Serial port (default: {PORT})")
    parser.add_argument("--monitor", action="store_true",
                        help="Chế độ debug: in thô serial ra màn hình")
    args = parser.parse_args()

    if args.monitor:
        monitor(args.port)
    elif args.env is None or args.label is None:
        parser.error("--env và --label bắt buộc khi không dùng --monitor")
    else:
        collect(args.env, args.label, args.n, args.port)


if __name__ == "__main__":
    main()
