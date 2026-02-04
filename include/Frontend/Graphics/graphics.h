// Copyright 2026 Robert Carneiro, Derek Meer, Matthew Tabak, Eric Lujan
//
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
#ifndef _GRAPHICS
#define _GRAPHICS

#include "Backend/Database/GString.h"
#ifdef GFX_HAVE_SDL2
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <SDL2/SDL_ttf.h>
#endif
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <map>
#include <string>
#include "GfxTypes.h"

// Forward declaration to avoid heavy GLFW include in header
struct GLFWwindow;

class GItem;
class RUComponent;
class RULabel;
class GPanel;
class Object;
class GFont;
class GfxRenderer;
#ifdef GFX_HAVE_OPENGL
class GLTextRenderer;
#endif

class gfxpp
{
	friend class RUComponent;
	friend class RUTextComponent;
	friend class RUImageComponent;
	friend class RUButton;
	friend class RULabel;
	friend class RUTextbox;
	friend class RUGraph;
	friend class RUProgressBar;
	friend class RUScrollbar;
	friend class RUListbox;
	friend class RUDropdown;
	friend class RUKeyDown;
	friend class RUKeyUp;

	// GLFW callback helpers need access to private state
	friend void glfw_window_close_cb(GLFWwindow*);
	friend void glfw_key_cb(GLFWwindow*, int, int, int, int);
	friend void glfw_mouse_button_cb(GLFWwindow*, int, int, int);
	friend void glfw_cursor_pos_cb(GLFWwindow*, double, double);
	friend void glfw_scroll_cb(GLFWwindow*, double, double);
	friend void glfw_framebuffer_size_cb(GLFWwindow*, int, int);

public:
	enum RenderBackend
	{
		RENDER_BACKEND_SDL2 = 0,
		RENDER_BACKEND_OPENGL = 1
	};

private:
	int errorFlag;
	bool running;
	int width;
	int height;
	float hunterZolomon; // zoom

	int32_t frames;
	float fps;
	bool rotate;
	bool move;
	int32_t now;
	int32_t then;

	// for mouse
	int mouseX;
	int mouseY;

	// for key presses
	bool CTRLPressed;
	bool ALTPressed;
	bool spacePressed;
	bool fPressed;
	bool uPressed;
	bool qPressed;
	bool gPressed;
	bool rPressed;
	bool lPressed;
	bool upPressed;
	bool downPressed;
	bool leftPressed;
	bool rightPressed;

	#ifdef GFX_HAVE_SDL2
	SDL_Window* window;
	SDL_GLContext context;
	SDL_Renderer* renderer;
	#endif
	GLFWwindow* glfwWindow;
	bool glfwInitialized;
	RenderBackend renderBackend;
	GfxRenderer* draw;
	bool ttfReady;
	bool finalized;

	std::vector<GItem*> guiElements; // < RUComponent* || GLayout* >

	#ifdef GFX_HAVE_SDL2
	SDL_Cursor* systemCursor;
	#endif
	GItem* focusedItem;

	// External event listeners use unified GfxEvent (SDL-backed or stubbed)
	typedef void (*EventListenerFn)(const GfxEvent&, void*);
	struct EventListener { EventListenerFn fn; void* userData; };
	std::vector<EventListener> listeners;

	// main
	void display();
	int initHelper(bool, shmea::GString, bool);
	int init2D(bool);
	int initOpenGL();
	void clean2D();

	// For GLFW-based OpenGL path: synthetic event queue (GfxEvent container)
	std::vector<GfxEvent> glfwEventQueue;

	// OpenGL text cache (fontPath+size -> renderer)
#ifdef GFX_HAVE_OPENGL
	std::map<std::string, GLTextRenderer*> glTextCache;
#endif

public:
	static const float MAX_FRAMES_PER_SECOND;

	static const int X_AXIS = 0;
	static const int Y_AXIS = 1;
	static const int Z_AXIS = 2;

	//TODO: Remove this in future since we decoupled the 2D and 3D rendering
	static const int _2D = 0;

	gfxpp();
	gfxpp(shmea::GString, int = _2D, bool = true, bool = true, int = 800, int = 600);
	// External window constructors
	gfxpp(GLFWwindow* externalWindow, int width, int height);
	#ifdef GFX_HAVE_SDL2
	gfxpp(SDL_Window* externalWindow, SDL_Renderer* externalRenderer, int width, int height);
	#endif
	~gfxpp();
	int getErrorFlag() const;
	#ifdef GFX_HAVE_SDL2
	SDL_Renderer* getRenderer();
	#endif
	GfxRenderer* getDraw() { return draw; }
	RenderBackend getRenderBackend() const { return renderBackend; }

	// GFX Utils
	static unsigned int RGBfromHue(double, int8_t*, int8_t*, int8_t*);

	// 2D
	GFont* cFont;
	std::map<int, GFont*> graphicsFonts;

	GPanel* focusedPanel;
	void addGradient(int, int, int);
	void addItem(GItem*);
	void removeItem(int); // id
	GItem* getItemByID(int);
	void setFocus(GItem*);
	GfxCursor* getSystemCursor();
	void setCursor(GfxCursor*);
	int getWidth() const;
	int getHeight() const;

	// main
	void run();
	void finish();
	// Request that the gfx loop exit at the next safe point. This only clears
	// the running flag; cleanup still happens in finish(), which is typically
	// called after run()/display() return.
	void requestFinish() { running = false; }
	bool getRunning() const;

	// Get a cached GL text renderer for a font path and pixel height; created on demand
#ifdef GFX_HAVE_OPENGL
	GLTextRenderer* getGLText(const std::string& fontPath, int pixelHeight);
#endif

	// Event listeners (available for both SDL2 and OpenGL builds)
	void addEventListener(EventListenerFn fn, void* userData) { if(fn) listeners.push_back((EventListener){fn, userData}); }
};

#endif
