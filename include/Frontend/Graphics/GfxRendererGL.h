#ifndef _GFX_RENDERER_GL_H
#define _GFX_RENDERER_GL_H

#include "GfxRenderer.h"

// Ensure OpenGL symbols are available regardless of inclusion order
#ifdef GFX_HAVE_SDL2
#include <SDL.h>
#include <SDL_opengl.h>
#else
#ifdef GFX_HAVE_OPENGL
#include <GL/gl.h>
#endif
#endif

class GfxRendererGL : public GfxRenderer
{
public:
	GfxRendererGL(SDL_Window* w, SDL_GLContext c) : sdlWindow(w), glContext(c), r(0), g(0), b(0), a(255) {}
	~GfxRendererGL() {}

	void setDrawColor(Uint8 rr, Uint8 gg, Uint8 bb, Uint8 aa) { r = rr; g = gg; b = bb; a = aa; glColor4ub(r, g, b, aa); }
	void clear() { glClearColor(r/255.0f, g/255.0f, b/255.0f, a/255.0f); glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); }
	void present() { SDL_GL_SwapWindow(sdlWindow); }
	void setLogicalSize(int width, int height) {
		int fbW = width, fbH = height;
		if (sdlWindow) {
			SDL_GL_GetDrawableSize(sdlWindow, &fbW, &fbH);
		}
		glViewport(0, 0, fbW, fbH);
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho(0.0, (GLdouble)width, (GLdouble)height, 0.0, -1.0, 1.0);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
	}

	void drawPoint(int x, int y) { glDisable(GL_TEXTURE_2D); glBegin(GL_POINTS); glVertex2i(x, y); glEnd(); }
	void drawPoints(const std::vector<GfxPoint>& points) { if(points.empty()) return; glDisable(GL_TEXTURE_2D); glBegin(GL_POINTS); for(size_t i=0;i<points.size();++i){ glVertex2i(points[i].x, points[i].y);} glEnd(); }
	void drawLine(int x1, int y1, int x2, int y2) { glDisable(GL_TEXTURE_2D); glBegin(GL_LINES); glVertex2i(x1,y1); glVertex2i(x2,y2); glEnd(); }
	void drawLines(const std::vector<GfxPoint>& points) { if(points.size()<2) return; glDisable(GL_TEXTURE_2D); glBegin(GL_LINE_STRIP); for(size_t i=0;i<points.size();++i){ glVertex2i(points[i].x, points[i].y);} glEnd(); }
	void drawRect(const GfxRect* rect) { if(!rect) return; glDisable(GL_TEXTURE_2D); glBegin(GL_LINE_LOOP); glVertex2i(rect->x, rect->y); glVertex2i(rect->x+rect->w, rect->y); glVertex2i(rect->x+rect->w, rect->y+rect->h); glVertex2i(rect->x, rect->y+rect->h); glEnd(); }
	void fillRect(const GfxRect* rect) { if(!rect) return; glDisable(GL_TEXTURE_2D); glBegin(GL_QUADS); glVertex2i(rect->x, rect->y); glVertex2i(rect->x+rect->w, rect->y); glVertex2i(rect->x+rect->w, rect->y+rect->h); glVertex2i(rect->x, rect->y+rect->h); glEnd(); }

	void copyTexture(GfxTexture*, const GfxRect*, const GfxRect*) { /* Not implemented for GL path */ }
	GfxTexture* createRenderTargetTexture(int, int) { return NULL; }
	void setTargetTexture(GfxTexture*) {}
	void resetTarget() {}
	void setTextureBlendMode(GfxTexture*, int) {}
	GfxTexture* createTextureFromSurface(GfxSurface*) { return NULL; }

private:
	SDL_Window* sdlWindow;
	SDL_GLContext glContext;
	Uint8 r,g,b,a;
};

#endif


