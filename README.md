# DELTAR — xuất depth gốc từ RGB và ToF 8×8

Hướng dẫn này dùng Logitech C270 và VL53L8CX qua ESP32 trên Windows. Chế độ chạy dùng **đầu ra gốc của DELTAR**: tắt lọc ToF theo thời gian, làm mượt depth và trọng số biên RGB. Cửa sổ có hai cột RGB và depth, mô hình suy luận một lần mỗi khung để giảm tải. Hiệu chuẩn camera và phép chiếu ToF vẫn được áp dụng.

## 1. Cài đặt trên máy người nhận

Mở PowerShell tại thư mục chứa file README này. Ví dụ:

```powershell
Set-Location 'G:\LUMI\ToF_camera\deltar\deltar'
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

Cài cặp PyTorch/torchvision phù hợp theo [hướng dẫn PyTorch](https://pytorch.org/get-started/locally/). Dùng bản CUDA với GPU NVIDIA để ưu tiên FPS. Thay phần `pip` trong lệnh của PyTorch bằng `.\.venv\Scripts\python.exe -m pip` để cài đúng môi trường. Ví dụ cho máy chỉ chạy CPU:

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Sau đó cài các thư viện còn lại và kiểm tra CUDA:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

Trọng số đã có tại `weights/deltar_nyu.pt`. Nếu chưa nối phần cứng, có thể kiểm tra việc nạp trọng số và suy luận:

```powershell
.\.venv\Scripts\python.exe -m apps.live_depth --self-test --frames 3 --extrinsics configs/calibration/rgb_tof_extrinsics_25mm.json --smoothing off --no-edge-aware-tof --view clean
```

Kết quả mong đợi: `Weights matched`, `prediction_shape: [480, 640]` và `all_finite: true`.

## 2. Chạy depth gốc với camera và ToF

Nối camera và ESP32. Camera cần trả ảnh **640×480**; ESP32 cần có firmware xuất frame ToF F hoặc F2 ở **921600 baud**. Kiểm tra cổng COM trong Device Manager và đóng Serial Monitor hoặc ứng dụng đang giữ camera/cổng COM. Ví dụ dưới dùng camera **1** và **COM9**; thay bằng giá trị của máy đang chạy.

Kiểm tra thiết bị:

```powershell
.\.venv\Scripts\python.exe -m apps.check_camera_tof --camera 1 --port COM9 --baud 921600
```

Chạy bản depth gốc:

```powershell
.\scripts\windows\run_calibrated_depth.bat --camera 1 --port COM9
```

Hoặc chạy lệnh Python tương đương:

```powershell
.\.venv\Scripts\python.exe -m apps.live_depth --camera 1 --port COM9 --baud 921600 --sigma-m 0.02 --extrinsics configs/calibration/rgb_tof_extrinsics_25mm.json --smoothing off --no-edge-aware-tof --view clean
```

Cửa sổ hiển thị RGB đã khử méo và `Depth - off` (`off` nghĩa là tắt bộ lọc). Nhấn **Q** hoặc **Esc** để thoát. Chương trình tự dùng CUDA nếu có.

## 3. Lưu depth gốc để dựng 3D

Lệnh sau chạy không mở cửa sổ xem, xử lý 30 khung và lưu **khung cuối**:

```powershell
.\scripts\windows\run_calibrated_depth.bat --camera 1 --port COM9 --headless --frames 30 --save-dir outputs/raw_depth
```

| File trong `outputs/raw_depth/` | Cách dùng |
|---|---|
| `last_depth_m.npy` | Mảng depth float32, 480 hàng × 640 cột, đơn vị **mét**. Dùng file này để dựng hình. |
| `last_depth_mm.png` | Ảnh depth 16 bit, đơn vị **mm**. Đọc dưới dạng uint16 và chia 1000 để đổi sang mét. |
| `last_depth_color.png` | Ảnh màu để xem nhanh. |
| `summary.json` | Ghi chế độ chạy, số vùng ToF hợp lệ và thời gian xử lý. |

Để dựng point cloud trong hệ tọa độ camera, dùng ma trận `camera.K` trong đúng file JSON truyền vào `--extrinsics`. Với pixel `(u, v)` và depth `Z` theo mét: `X = (u - cx) * Z / fx`, `Y = (v - cy) * Z / fy`. Depth nằm trên lưới ảnh **đã khử méo 640×480**. Lệnh lưu hiện chỉ xuất snapshot depth, không lưu ảnh RGB hay chuỗi khung.

Thông số hiệu chuẩn đi kèm là ước lượng của cụm camera–ToF hiện tại. Khi đổi camera hoặc cách gá, cần hiệu chuẩn lại. Kiểm tra tỷ lệ và khoảng cách bằng vật có kích thước đã biết trước khi dùng bản dựng để đo.

Mã nguồn gốc: [DELTAR (ECCV 2022)](https://zju3dv.github.io/deltar/). Giữ file `LICENSE` khi chia sẻ.
