// OpenGL texture wrapper for builds without SDL2.
#ifndef _GFX_TEXTURE_GL_H
#define _GFX_TEXTURE_GL_H

#include "GfxTypes.h"

#if defined(GFX_HAVE_OPENGL) && !defined(GFX_HAVE_SDL2)
#include <GLFW/glfw3.h>

// When SDL2 is disabled, GfxTypes.h defines GfxTexture as an opaque struct.
// This header provides its concrete OpenGL-backed definition.
struct GfxTextureOpaque
{
	GLuint textureId;
	GLuint fboId; // 0 when not a render target
	int width;
	int height;
	int blendMode;
	bool isRenderTarget;

	GfxTextureOpaque()
		: textureId(0),
		  fboId(0),
		  width(0),
		  height(0),
		  blendMode(GFX_BLENDMODE_BLEND),
		  isRenderTarget(false)
	{
	}
};

#endif

#endif

