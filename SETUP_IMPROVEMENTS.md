# EdgeVisionRT Setup Improvements

## Ngày cập nhật: 5 Tháng 1, 2026

## Các thay đổi chính trong setup.sh

### 1. Build NCNN đúng cách với đầy đủ cmake config
- **Trước**: NCNN được build không có cmake config files, gây lỗi khi project tìm ncnnConfig.cmake
- **Sau**: Build và install NCNN đúng cách với:
  - CMAKE_INSTALL_PREFIX để tạo cấu trúc thư mục đúng
  - Đầy đủ lib/cmake/ncnn/ncnnConfig.cmake
  - Hỗ trợ Vulkan GPU acceleration
  - Hỗ trợ INT8 quantization (tăng tốc 2-4x)

### 2. Kiểm tra và xác minh installation
- Kiểm tra sự tồn tại của ncnnConfig.cmake trước khi build
- Tự động xóa các installation không đầy đủ
- Xác minh sau khi install xong

### 3. Error handling tốt hơn
- Kiểm tra exit code của mỗi bước (cmake, make, make install)
- Hiển thị thông báo lỗi rõ ràng
- Dừng ngay khi có lỗi (set -e)

### 4. Tối ưu hóa build
- Sử dụng --depth=1 khi clone để nhanh hơn
- Build với -j$(nproc) để tận dụng tất cả CPU cores
- Bật NCNN_INT8 để có khả năng quantization

## Các file đã được fix

### CMakeLists.txt
- Thêm highgui component vào OpenCV
- Cấu hình warning flags hợp lý
- Loại bỏ các conversion warnings không cần thiết

### Source code fixes
1. **src/neon_preprocess.cpp**: 
   - Loại bỏ unused variables (v_wy0, v_wy1, letterbox_center)
   - Đánh dấu unused function với __attribute__((unused))

2. **src/input_pipeline.cpp**:
   - Comment out unused variables (video_width, video_height)

3. **src/main.cpp**:
   - Đổi tên biến để tránh shadowing (err -> infer_err)

4. **include/video_writer.h**:
   - Sắp xếp lại thứ tự khởi tạo member variables trong constructor
   - AsyncVideoWriter: ffmpeg_pipe_ -> running_ -> frames_written_ -> frames_dropped_
   - AsyncDisplay: display_width_ -> display_height_ -> running_ -> frames_displayed_ -> frames_dropped_ -> pending_frame_ready_

## Kết quả

✅ **0 warnings** khi build với flags nghiêm ngặt:
   - `-Wall -Wextra -Wpedantic -Wshadow -Wnull-dereference -Wformat=2`

✅ **0 errors** 

✅ Build thành công với:
   - NCNN Vulkan: ON
   - NCNN INT8: ON  
   - OpenCV 4.6.0 với highgui
   - Full optimization flags cho RPi 5

## Cách sử dụng setup.sh mới

```bash
# Chạy setup (cần sudo cho system dependencies)
sudo ./setup.sh

# Sau đó build project
./build.sh

# Hoặc chạy luôn
./run.sh
```

## Lưu ý
- Script tự động phát hiện nếu NCNN đã được cài đặt đúng
- Nếu phát hiện installation không đầy đủ, sẽ tự động rebuild
- Quá trình build NCNN mất khoảng 5-10 phút trên RPi 5
