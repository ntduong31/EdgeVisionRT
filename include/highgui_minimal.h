/**
 * @file highgui_minimal.h
 * @brief Minimal highgui declarations (when dev package not available)
 */

#ifndef HIGHGUI_MINIMAL_H
#define HIGHGUI_MINIMAL_H

#include <opencv2/core.hpp>
#include <string>

namespace cv {

// Window flags
enum WindowFlags {
    WINDOW_NORMAL     = 0x00000000,
    WINDOW_AUTOSIZE   = 0x00000001,
    WINDOW_OPENGL     = 0x00001000,
    WINDOW_FULLSCREEN = 1,
    WINDOW_FREERATIO  = 0x00000100,
    WINDOW_KEEPRATIO  = 0x00000000,
    WINDOW_GUI_EXPANDED = 0x00000000,
    WINDOW_GUI_NORMAL = 0x00000010
};

// Window functions - these are defined in libopencv_highgui.so
void namedWindow(const std::string& winname, int flags = WINDOW_AUTOSIZE);
void destroyWindow(const std::string& winname);
void destroyAllWindows();
void imshow(const std::string& winname, InputArray mat);
int waitKey(int delay = 0);
void resizeWindow(const std::string& winname, int width, int height);
void moveWindow(const std::string& winname, int x, int y);
void setWindowTitle(const std::string& winname, const std::string& title);

} // namespace cv

#endif // HIGHGUI_MINIMAL_H
