#ifndef _GFX_RENDERER_GLFW_H
#define _GFX_RENDERER_GLFW_H

#include "GfxRenderer.h"
#include <GLFW/glfw3.h>

class GfxRendererGLFW : public GfxRenderer
{
public:
	GfxRendererGLFW(GLFWwindow* w);
	~GfxRendererGLFW();

	// Frame
	void setDrawColor(Uint8 rr, Uint8 gg, Uint8 bb, Uint8 aa);
	void clear();
	void present();
	void setLogicalSize(int width, int height);

	// Primitives
	void drawPoint(int x, int y);
	void drawPoints(const std::vector<GfxPoint>& points);
	void drawLine(int x1, int y1, int x2, int y2);
	void drawLines(const std::vector<GfxPoint>& points);
	void drawRect(const GfxRect* rect);
	void fillRect(const GfxRect* rect);

	// Textures / targets
	void copyTexture(GfxTexture* texture, const GfxRect* src, const GfxRect* dst);
	GfxTexture* createRenderTargetTexture(int width, int height);
	void setTargetTexture(GfxTexture* texture);
	void resetTarget();
	void setTextureBlendMode(GfxTexture* texture, int mode);
	GfxTexture* createTextureFromSurface(GfxSurface* surface);

private:
	GLFWwindow* window;
	Uint8 r,g,b,a;
	int logicalW;
	int logicalH;

#if defined(GFX_HAVE_OPENGL) && !defined(GFX_HAVE_SDL2)
	// Best-effort support for EXT/core FBO entry points
	void* p_glGenFramebuffers;
	void* p_glBindFramebuffer;
	void* p_glFramebufferTexture2D;
	void* p_glCheckFramebufferStatus;
	void* p_glDeleteFramebuffers;

	GfxTexture* currentTarget;
#endif
};

#endif


