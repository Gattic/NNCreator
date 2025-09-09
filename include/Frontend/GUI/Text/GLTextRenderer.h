#ifndef _GL_TEXT_RENDERER_H
#define _GL_TEXT_RENDERER_H

#include <ft2build.h>
#include FT_FREETYPE_H
#include <map>
#include <string>
#ifdef GFX_HAVE_OPENGL
#include <GLFW/glfw3.h>
#endif

struct GLGlyph {
	GLuint textureId;
	int width;
	int height;
	int bearingX;
	int bearingY;
	int advance;
};

class GLTextRenderer {
public:
	GLTextRenderer();
	~GLTextRenderer();
	bool init(const std::string& fontPath, int pixelHeight);
	bool initOrUpdate(const std::string& fontPath, int pixelHeight);
	void drawText(float x, float y, const std::string& text, unsigned char r, unsigned char g, unsigned char b, unsigned char a);
	// Measure total advance width of a string (in pixels)
	int measureTextWidth(const std::string& text) const;
	// Advance width for a single glyph (in pixels); returns 0 if glyph missing
	int getGlyphAdvance(char c) const;

private:
	bool initialized;
	FT_Library ft;
	FT_Face face;
	std::map<char, GLGlyph> glyphs;
	std::string fontPathUsed;
	int pixelHeightUsed;
	int maxBearingY; // for top alignment
};

#endif


