# Báo cáo Chuyên đề Nghiên cứu 1
## ECA-UWB: Mạng hai nhánh nhẹ với Efficient Channel Attention cho nhận dạng NLOS trong UWB

**Sinh viên:** Nguyễn Tấn Phong  
**Giảng viên hướng dẫn:** [Tên giáo sư]  
**Đơn vị:** Khoa Điện tử Viễn thông, Trường ĐHCN – ĐHQGHN  
**Thời gian:** Tháng 3/2026

---

## 1. Giới thiệu bài toán

### 1.1. UWB và bài toán định vị trong nhà

Ultra-Wideband (UWB) là công nghệ truyền thông không dây sử dụng băng thông rất rộng (>500 MHz), cho phép đo khoảng cách với độ chính xác cấp centimet thông qua phương pháp Time-of-Arrival (TOA). Trong hệ thống định vị UWB, một **tag** (thiết bị gắn trên đối tượng cần định vị) giao tiếp với nhiều **anchor** (điểm neo cố định có tọa độ biết trước) để tính vị trí bằng thuật toán trilateration.

### 1.2. Vấn đề NLOS (Non-Line-of-Sight)

Trong môi trường thực tế (nhà máy, bệnh viện, kho hàng), tín hiệu UWB thường bị chặn bởi tường, trụ bê tông, thiết bị kim loại. Khi đó, tín hiệu đi theo đường phản xạ/nhiễu xạ thay vì đường thẳng trực tiếp. Điều này tạo ra **sai số dương** (positive bias) trong phép đo khoảng cách: khoảng cách đo được luôn lớn hơn khoảng cách thực, gây sai lệch nghiêm trọng cho kết quả định vị.

**Tại sao cần nhận dạng NLOS?** Nếu biết anchor nào đang ở trạng thái NLOS, hệ thống có thể loại bỏ hoặc hiệu chỉnh phép đo đó trước khi tính vị trí. Do đó, **NLOS identification** (phân loại LOS/NLOS) là bước tiên quyết cho định vị chính xác.

### 1.3. Channel Impulse Response (CIR)

CIR là tín hiệu phản hồi kênh truyền, được chip UWB DW1000 (Decawave) ghi lại dưới dạng 1,016 mẫu. CIR mã hóa cấu trúc đa đường (multipath) của kênh truyền:
- **LOS CIR**: Đỉnh first-path mạnh, sắc nét
- **NLOS CIR**: Đỉnh first-path yếu, bị trễ, bị "smear" (trải rộng)

Ngoài CIR, DW1000 cung cấp 14 thanh ghi **channel diagnostics** chứa các thông số thống kê của tín hiệu (biên độ first-path, nhiễu, công suất CIR...).

---

## 2. Tổng quan các phương pháp hiện có (Related Work)

### 2.1. Phương pháp truyền thống

Các phương pháp đầu tiên trích xuất đặc trưng thống kê từ CIR (kurtosis, RMS delay spread, tỷ lệ biên độ first-path) rồi dùng SVM hoặc k-NN để phân loại. Độ chính xác đạt 75–82%.

### 2.2. Phương pháp Deep Learning

| Phương pháp | Năm | Accuracy | Số tham số | Đặc điểm |
|---|---|---|---|---|
| CNN (Bregar) | 2018 | 87.38% | ~50K | Baseline đầu tiên trên eWINE |
| CNN-LSTM | 2020 | 88.82% | 7,441 | Kết hợp CNN trích đặc trưng không gian + LSTM cho chuỗi thời gian |
| FCN-Attention | 2024 | 88.24% | ~200K | Multi-head self-attention trên toàn bộ CIR |
| CIR-CNN+MLP (Si) | 2023 | 87.96% | 1,578 | Lightweight, tránh "data inundation" |
| SA-TinyML (Wu) | 2024 | 93.65% | 4,627 | 2-stage pipeline, self-attention, SOTA |
| LightMamba (Wang) | 2026 | 92.38% | 25,793 | Selective state-space model (Mamba) |
| MS-CNN-SA (Jiang) | 2026 | 93.10% | 24,145 | Multi-scale CNN + 8-head self-attention |

### 2.3. Hai khoảng trống nghiên cứu (Research Gaps)

**Gap 1:** Các cơ chế attention hiện có (self-attention, SE-Net) đều tốn hàng trăm đến hàng nghìn tham số. ECA (Efficient Channel Attention) chỉ cần k tham số nhưng chưa ai áp dụng cho NLOS classification trên benchmark chuẩn.

**Gap 2:** Hầu hết các bài báo chỉ test trên 1 dataset (eWINE), không kiểm tra liệu mô hình có hoạt động được ở môi trường khác (cross-environment generalization).

---

## 3. Phương pháp đề xuất: ECA-UWB

### 3.1. Tổng quan kiến trúc

ECA-UWB là mạng hai nhánh (dual-branch):
- **Nhánh CIR**: Xử lý 50 mẫu CIR bằng 2 lớp Conv1d với ECA module ở giữa
- **Nhánh Auxiliary**: Xử lý 7 đặc trưng channel diagnostics bằng MLP 2 lớp
- **Gated Fusion**: Kết hợp 2 nhánh bằng cổng học được (learned gates)
- **Classification Head**: Phân loại nhị phân LOS/NLOS

**Input:** Vector 57 chiều = [50 mẫu CIR + 7 diagnostics]  
**Output:** Logit ŷ, phân loại NLOS nếu σ(ŷ) > τ = 0.54

### 3.2. Giải thích các thành phần AI

#### 3.2.1. Conv1d (1-D Convolution)

**Ý tưởng đơn giản:** Giống như một cửa sổ trượt (sliding window) chạy dọc theo tín hiệu CIR. Tại mỗi vị trí, cửa sổ nhân tín hiệu với một bộ trọng số (kernel) rồi cộng lại thành 1 giá trị. Kết quả là một "bản đồ đặc trưng" (feature map) nắm bắt các pattern cục bộ trong CIR.

**Trong ECA-UWB:**
- Conv1 (1→16, k=5): 1 kênh input → 16 kênh output, mỗi kernel nhìn 5 mẫu liên tiếp. Tức là tạo 16 "bộ lọc" khác nhau, mỗi bộ lọc phát hiện 1 loại pattern trong CIR.
- Conv2 (16→16, k=3): 16 kênh → 16 kênh, mỗi kernel nhìn 3 mẫu.

#### 3.2.2. Batch Normalization (BN)

Chuẩn hóa output của mỗi lớp Conv để có mean ≈ 0, std ≈ 1. Giúp training ổn định hơn và nhanh hội tụ.

#### 3.2.3. ReLU (Activation Function)

Hàm kích hoạt: ReLU(x) = max(0, x). Biến đổi phi tuyến, cho phép mạng học các pattern phức tạp. Tất cả giá trị âm bị đặt về 0.

#### 3.2.4. MaxPool (Max Pooling)

Giảm kích thước feature map bằng cách lấy giá trị lớn nhất trong mỗi cửa sổ. MaxPool(2) giảm chiều dài từ 50 → 25 mẫu.

#### 3.2.5. GAP (Global Average Pooling)

Lấy trung bình toàn bộ chiều thời gian. Biến feature map (B, 16, 25) → vector (B, 16). Mỗi giá trị đại diện cho "mức năng lượng" trung bình của 1 kênh.

#### 3.2.6. ECA Module (Efficient Channel Attention)

**Đây là đóng góp cốt lõi của bài báo.** ECA quyết định "kênh nào quan trọng" và tăng/giảm trọng số tương ứng:

1. **GAP**: Nén feature map → vector C chiều (C=16)
2. **Transpose + Conv1d(k=3)**: Mỗi kênh "nhìn" 2 kênh lân cận. Ví dụ: kênh thứ 5 được tính dựa trên kênh 4, 5, 6.
3. **Sigmoid**: Tạo trọng số w ∈ (0, 1) cho mỗi kênh
4. **Channel-wise multiply**: F' = w ⊙ F

**So sánh với SE-Net:** SE-Net dùng 2 lớp fully-connected (FC) → 256 tham số (C=16, r=2). ECA chỉ dùng 1 Conv1d(k=3) → **3 tham số**. Giảm 85× số tham số!

**Tại sao ECA phù hợp cho CIR?** Các filter CNN lân cận thường bắt các pattern tương tự (ví dụ: filter 5 bắt "rising edge", filter 6 bắt "peak"). Conv1d(k=3) mô hình hóa tương tác cục bộ này một cách tự nhiên.

#### 3.2.7. Gated Feature Fusion

Thay vì ghép nối đơn giản (concatenation) hai vector f_cir và f_aux, ECA-UWB học hai "cổng" g₁, g₂ cho mỗi mẫu:

```
g = σ(W_g [f_cir; f_aux] + b_g) ∈ R²
f_fused = g₁ · f_cir + g₂ · f_aux
```

Ý nghĩa: Nếu tín hiệu CIR rõ ràng (LOS mạnh), mô hình tự động tăng g₁ (tin vào CIR). Nếu CIR nhiễu nhưng diagnostics cho thấy NLOS rõ, mô hình tăng g₂.

#### 3.2.8. Cost-Sensitive Training

Sử dụng hàm mất mát BCE có trọng số: w⁺=1.5. Tức là phạt false negative (bỏ sót NLOS) nặng hơn 1.5 lần so với false positive. Lý do: bỏ sót NLOS gây sai số định vị lớn hơn so với phát hiện nhầm LOS thành NLOS.

### 3.3. Tổng số tham số

| Thành phần | Tham số | % |
|---|---|---|
| Conv1 + BN | 128 | 5.4% |
| **ECA** | **3** | **0.1%** |
| Conv2 + BN | 816 | 34.4% |
| Aux MLP | 784 | 33.0% |
| Gated Fusion | 66 | 2.8% |
| Classifier | 577 | 24.3% |
| **Tổng** | **2,374** | **100%** |

---

## 4. Thiết lập thí nghiệm

### 4.1. Dataset 1: eWINE Benchmark

- 42,000 mẫu UWB thu từ 7 môi trường indoor tại châu Âu
- Hardware: Decawave DW1000
- Balanced: 21,000 LOS + 21,000 NLOS
- Split: 70/15/15 (train/val/test), seed=42
- Preprocessing: Cắt 50 mẫu CIR quanh first-path, chuẩn hóa

### 4.2. Dataset 2: VNU Indoor

- Thu thập tại ĐH Công nghệ – ĐHQGHN
- Hardware: Decawave DWM1001 (khác hardware eWINE)
- 3 môi trường: Phòng thực hành, Hành lang, Sảnh giảng đường
- 9,000 mẫu (3×1,500 LOS + 3×1,500 NLOS)
- Khoảng cách tag-anchor: 1–10 m

### 4.3. Cross-dataset Protocol

- **Zero-shot**: Dùng trực tiếp model eWINE để test trên VNU, không fine-tune
- **Head-adapted**: Đóng băng backbone, chỉ fine-tune 643 params (gate + classifier) trên 20% dữ liệu VNU

---

## 5. Kết quả

### 5.1. Kết quả trên eWINE

| Metric | ECA-UWB | SA-TinyML | LightMamba |
|---|---|---|---|
| Accuracy | 91.45% | **93.65%** | 92.38% |
| NLOS Recall | 91.44% | **93.10%** | — |
| AUC-ROC | 97.48% | **98.22%** | — |
| Params | **2,374** | 4,627 | 25,793 |
| Latency | 0.57 ms | 0.64 ms | 1.61 ms |

**Phân tích:** ECA-UWB chênh 2.20 pp so với SA-TinyML nhưng ít hơn 49% tham số và chỉ cần single-stage training (4.6 phút). So với LightMamba, chênh chỉ 0.93 pp nhưng ít hơn **10.9×** tham số.

### 5.2. Ablation Study

| Variant | Accuracy | ΔAcc | Params |
|---|---|---|---|
| Bỏ Aux Branch | 86.47% | **-5.65 pp** | 1,524 |
| Bỏ ECA | 91.82% | -0.30 pp | 2,371 |
| Dùng Concat thay Gate | 92.02% | -0.10 pp | 2,836 |
| **Full (proposed)** | **92.12%** | — | **2,374** |

**Kết luận:** Nhánh auxiliary (7 diagnostics) đóng góp lớn nhất (+5.65 pp). ECA thêm 0.30 pp chỉ với 3 params. Gated fusion tốt hơn concatenation và nhẹ hơn 462 params.

### 5.3. Cross-dataset (eWINE → VNU)

| Phương pháp | Zero-shot Acc | AUC | Head-adapted Acc |
|---|---|---|---|
| SA-TinyML | **90.24%** | **0.968** | 93.14% |
| MS-CNN-SA | 68.12% | 0.958 | 89.92% |
| **ECA-UWB** | 80.36% | 0.942 | **98.21%** |

**Phân tích:** ECA-UWB có zero-shot accuracy thấp hơn SA-TinyML (80% vs 90%), nhưng AUC vẫn cao (0.942), chứng tỏ features vẫn phân biệt được LOS/NLOS — chỉ là threshold chưa chuẩn. Sau khi fine-tune chỉ 643 params trên 20% dữ liệu VNU, ECA-UWB đạt **98.21%** — cao nhất, vượt SA-TinyML 5.07 pp. Điều này chứng tỏ backbone features của ECA-UWB transferable nhất.

---

## 6. Kết luận

ECA-UWB đạt được sự cân bằng tốt nhất giữa accuracy và model size:
- **2,374 tham số** (9.5 KB float32), đủ nhỏ cho MCU STM32F4 (256 KB RAM)
- **91.45% accuracy** trên eWINE, single-stage training trong 4.6 phút
- **98.21% accuracy** khi head-adapted sang VNU dataset — kết quả cross-dataset tốt nhất
- Ablation study xác nhận vai trò của từng thành phần

**Target:** IEEE Sensors Journal (Q1, IF ≈ 4.3)

---

## 7. Hướng phát triển

1. **INT8 Post-Training Quantization**: Giảm model từ 9.5 KB → ~2.4 KB, deploy thật trên STM32 MCU
2. **Dual-task Formulation**: NLOS detection + ranging error estimation cùng lúc
3. **Domain Adaptation**: Tự động adapt sang hardware/environment mới với ít dữ liệu
4. **Tích hợp hệ thống**: Kết hợp ECA-UWB với Adaptive Kalman Filter (đã nghiên cứu ở ATC 2025) cho robot di động
