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
#ifndef _GFX_FONT_MANAGER_H
#define _GFX_FONT_MANAGER_H

#include "GfxTypes.h"
#include "Backend/Database/GPointer.h"
#include "../GUI/Text/GFont.h"
#include <map>
#include <string>

#ifdef GFX_HAVE_OPENGL
class GLTextRenderer;
#endif

class FontManager
{
private:
	shmea::GPointer<GFont> cFont;
	std::map<int, shmea::GPointer<GFont> > graphicsFonts;
#ifdef GFX_HAVE_OPENGL
	std::map<std::string, GLTextRenderer*> glTextCache;
#endif

public:
	FontManager();

	void initSDL(GfxNativeRenderer* sdlRenderer);
	void initOpenGL();
	void cleanup(bool isOpenGL);

	GFont* getDefaultFont();
	GFont* getFontByColor(int fontColor);

#ifdef GFX_HAVE_OPENGL
	GLTextRenderer* getGLText(const std::string& fontPath, int pixelHeight);
#endif
};

#endif
