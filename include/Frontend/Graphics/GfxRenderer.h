// Copyright 2026-2025 gfxplusplus contributors
// Minimal rendering abstraction to support SDL2 and OpenGL backends
#ifndef _GFX_RENDERER_H
#define _GFX_RENDERER_H

#include "GfxTypes.h"
#include <vector>

// Forward declaration to avoid heavy include
struct GLFWwindow;

class GfxRenderer
{
public:
	virtual ~GfxRenderer() {}

	// Frame
	virtual void setDrawColor(Uint8 r, Uint8 g, Uint8 b, Uint8 a) = 0;
	virtual void clear() = 0;
	virtual void present() = 0;
	virtual void setLogicalSize(int width, int height) = 0;

	// Primitives
	virtual void drawPoint(int x, int y) = 0;
	virtual void drawPoints(const std::vector<GfxPoint>& points) = 0;
	virtual void drawLine(int x1, int y1, int x2, int y2) = 0;
	virtual void drawLines(const std::vector<GfxPoint>& points) = 0; // polyline
	virtual void drawRect(const GfxRect* rect) = 0;
	virtual void fillRect(const GfxRect* rect) = 0;

	// Textures (SDL path only; OpenGL impl may be a no-op)
	virtual void copyTexture(GfxTexture* texture, const GfxRect* src, const GfxRect* dst) = 0;
	virtual GfxTexture* createRenderTargetTexture(int width, int height) = 0;
	virtual void setTargetTexture(GfxTexture* texture) = 0;
	virtual void resetTarget() = 0;
	virtual void setTextureBlendMode(GfxTexture* texture, int mode) = 0;
	virtual GfxTexture* createTextureFromSurface(GfxSurface* surface) = 0;

	// Factory helpers
	static GfxRenderer* createSDL(GfxNativeRenderer* sdlRenderer, SDL_Window* sdlWindow);
	static GfxRenderer* createOpenGL(SDL_Window* sdlWindow, SDL_GLContext glContext);
	static GfxRenderer* createOpenGL(GLFWwindow* glfwWindow);
};

#endif


