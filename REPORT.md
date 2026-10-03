# Lab 16 — Báo cáo: LightGBM trên AWS CPU Node

## Môi trường
- Hạ tầng: Terraform (VPC, Public/Private Subnet, NAT Gateway, Bastion `t3.micro`, ALB) — region `us-east-1`.
- Compute Node: **`c7i-flex.large`** (2 vCPU / 4 GB RAM), Ubuntu 22.04, Python 3.10, LightGBM 4.7.0.
- Thời gian `terraform apply` (từ lúc bắt đầu đến khi có Outputs): **~4 phút 10 giây**.
- Dataset: Kaggle Credit Card Fraud Detection — 284,807 giao dịch, 30 feature, 0.173% gian lận.
  Chia stratified 70% train / 10% validation (early stopping) / 20% test.

## Kết quả

| Metric | Kết quả |
|---|---|
| Thời gian load data | 0.940 s |
| Thời gian training | 3.142 s |
| Best iteration | 132 |
| AUC-ROC | 0.9746 |
| Accuracy | 0.9996 |
| F1-Score | 0.8619 |
| Precision | 0.9398 |
| Recall | 0.7959 |
| Inference latency (1 row) | 0.540 ms |
| Inference throughput (1000 rows) | 3.32 ms (~301,000 rows/s) |

## Nhận xét
1. **Training rất nhanh trên CPU:** chỉ ~3.1 s cho ~200k dòng × 30 feature trên 2 vCPU — với dữ liệu dạng bảng cỡ này, LightGBM không cần GPU; load CSV 144 MB cũng chưa tới 1 s.
2. **Chất lượng tốt:** AUC-ROC 0.975 trên tập test. Accuracy 99.96% không có nhiều ý nghĩa vì dữ liệu mất cân bằng (đoán toàn "không gian lận" đã đạt 99.83%), nên F1/Precision/Recall mới là chỉ số chính.
3. **Precision 0.94 / Recall 0.80** ở ngưỡng 0.5: model ít báo động nhầm nhưng bỏ sót ~20% giao dịch gian lận; có thể hạ ngưỡng để tăng Recall nếu bỏ sót gian lận tốn kém hơn báo nhầm.
4. **Inference rất nhanh:** ~0.5 ms/dòng (đủ cho real-time) và ~301k dòng/s khi chạy batch — 1 instance CPU nhỏ đủ phục vụ tải lớn.
5. **Tài nguyên:** CPU utilization chỉ đạt đỉnh ~15% (lúc cài thư viện và chạy benchmark), RAM dùng < 1 GB / 3.7 GB — instance còn dư nhiều tài nguyên.
6. **Chi phí:** toàn bộ hạ tầng khoảng $0.15/giờ (NAT Gateway là khoản lớn nhất cùng với compute node), được trừ vào Free Tier credit; đã `terraform destroy` ngay sau khi xong.

## Vấn đề gặp phải và cách xử lý
- **Free plan không cho chạy `t3.medium`** (`InvalidParameterCombination: not eligible for Free Tier`) → đổi sang `c7i-flex.large` (cùng 2 vCPU / 4 GB, thuộc Free Tier) qua `terraform/terraform.tfvars`.
- **Kaggle CLI 1.7.4 (Python 3.10) không đọc được API token mới dạng `KGAT_`** → tải dataset bằng `curl` gọi thẳng Kaggle API với header `Authorization: Bearer <token>`.
- **Lần chạy đầu `best_iteration = 1`, AUC chỉ 0.93:** do dữ liệu quá mất cân bằng, các lá có hessian rất nhỏ cho giá trị cực lớn làm AUC validation sụt ngay sau cây đầu tiên → thêm `min_child_weight=5`, `reg_lambda=5` → model học đủ 132 cây, AUC 0.975, F1 tăng từ 0.78 lên 0.86.
- **SSH từ Windows báo `UNPROTECTED PRIVATE KEY FILE`** → giới hạn quyền file `lab-key` bằng `icacls`.
