// Copyright 2026 Robert Carneiro, Derek Meer, Matthew Tabak, Eric Lujan
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
// associated documentation files (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge, publish, distribute,
// sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
// NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#ifndef _GFX_GLFW_EVENT_BRIDGE_H
#define _GFX_GLFW_EVENT_BRIDGE_H

#ifdef GFX_HAVE_OPENGL

#include "GfxTypes.h"
#include <vector>

struct GLFWwindow;

class GlfwEventBridge
{
private:
	std::vector<GfxEvent> eventQueue;
	int logicalW;
	int logicalH;
	bool closeRequested;

	static void windowCloseCb(GLFWwindow* w);
	static void keyCb(GLFWwindow* w, int key, int scancode, int action, int mods);
	static void mouseButtonCb(GLFWwindow* w, int button, int action, int mods);
	static void cursorPosCb(GLFWwindow* w, double x, double y);
	static void scrollCb(GLFWwindow* w, double xoffset, double yoffset);
	static void framebufferSizeCb(GLFWwindow* w, int ww, int hh);

public:
	GlfwEventBridge();

	void attach(GLFWwindow* window, int logicalWidth, int logicalHeight);
	void detach(GLFWwindow* window);
	void drainEvents(std::vector<GfxEvent>& out);
	bool isCloseRequested() const;
	void setLogicalSize(int w, int h);
};

#endif // GFX_HAVE_OPENGL
#endif
