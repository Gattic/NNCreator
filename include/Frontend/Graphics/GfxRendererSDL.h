#ifndef _GFX_RENDERER_SDL_H
#define _GFX_RENDERER_SDL_H

#include "GfxRenderer.h"

class GfxRendererSDL : public GfxRenderer
{
public:
	GfxRendererSDL(GfxNativeRenderer* r, SDL_Window* w) : sdlRenderer(r), sdlWindow(w) {}
	~GfxRendererSDL() {}

	void setDrawColor(Uint8 r, Uint8 g, Uint8 b, Uint8 a) { SDL_SetRenderDrawColor(sdlRenderer, r, g, b, a); }
	void clear() { SDL_RenderClear(sdlRenderer); }
	void present() { SDL_RenderPresent(sdlRenderer); }
	void setLogicalSize(int width, int height) { SDL_RenderSetLogicalSize(sdlRenderer, width, height); }

	void drawPoint(int x, int y) { SDL_RenderDrawPoint(sdlRenderer, x, y); }
	void drawPoints(const std::vector<GfxPoint>& points) { if(!points.empty()) SDL_RenderDrawPoints(sdlRenderer, (const SDL_Point*)&points[0], points.size()); }
	void drawLine(int x1, int y1, int x2, int y2) { SDL_RenderDrawLine(sdlRenderer, x1, y1, x2, y2); }
	void drawLines(const std::vector<GfxPoint>& points) { if(!points.empty()) SDL_RenderDrawLines(sdlRenderer, (const SDL_Point*)&points[0], points.size()); }
	void drawRect(const GfxRect* rect) { SDL_RenderDrawRect(sdlRenderer, (const SDL_Rect*)rect); }
	void fillRect(const GfxRect* rect) { SDL_RenderFillRect(sdlRenderer, (const SDL_Rect*)rect); }

	void copyTexture(GfxTexture* texture, const GfxRect* src, const GfxRect* dst) { SDL_RenderCopy(sdlRenderer, (SDL_Texture*)texture, (const SDL_Rect*)src, (const SDL_Rect*)dst); }
	GfxTexture* createRenderTargetTexture(int width, int height) { return (GfxTexture*)SDL_CreateTexture(sdlRenderer, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_TARGET, width, height); }
	void setTargetTexture(GfxTexture* texture) { SDL_SetRenderTarget(sdlRenderer, (SDL_Texture*)texture); }
	void resetTarget() { SDL_SetRenderTarget(sdlRenderer, NULL); }
	void setTextureBlendMode(GfxTexture* texture, int mode) { SDL_SetTextureBlendMode((SDL_Texture*)texture, (SDL_BlendMode)mode); }
	GfxTexture* createTextureFromSurface(GfxSurface* surface) { return (GfxTexture*)SDL_CreateTextureFromSurface(sdlRenderer, (SDL_Surface*)surface); }

private:
	GfxNativeRenderer* sdlRenderer;
	SDL_Window* sdlWindow;
};

#endif


