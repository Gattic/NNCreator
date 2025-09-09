// Unified graphics types for non-Graphics code. Internally backed by SDL when available.
#ifndef _GFX_TYPES_H
#define _GFX_TYPES_H

#include <stdint.h>

#ifdef GFX_HAVE_SDL2
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <SDL2/SDL_ttf.h>
#endif

// Basic fixed-size aliases used across code
typedef uint8_t Uint8;
typedef uint16_t Uint16;
typedef uint32_t Uint32;

#ifdef GFX_HAVE_SDL2
typedef SDL_Color GfxColor;
typedef SDL_Point GfxPoint;
typedef SDL_Rect GfxRect;
typedef SDL_Texture GfxTexture;
typedef SDL_Surface GfxSurface;
typedef TTF_Font TTF_Font_Type;
typedef SDL_Cursor GfxCursor;
typedef SDL_Event GfxEvent;
typedef SDL_Keycode GfxKeycode;
typedef SDL_SystemCursor GfxSystemCursor;
typedef SDL_Renderer GfxNativeRenderer;

// Byte order aliases
#define GFX_BYTEORDER SDL_BYTEORDER
#define GFX_BIG_ENDIAN SDL_BIG_ENDIAN

// Event/type aliases (avoid SDL_ in non-Graphics code)
#define GFX_MOUSEBUTTONDOWN SDL_MOUSEBUTTONDOWN
#define GFX_MOUSEBUTTONUP SDL_MOUSEBUTTONUP
#define GFX_MOUSEMOTION SDL_MOUSEMOTION
#define GFX_MOUSEWHEEL SDL_MOUSEWHEEL
#define GFX_KEYDOWN SDL_KEYDOWN
#define GFX_KEYUP SDL_KEYUP

#define GFX_ALPHA_OPAQUE SDL_ALPHA_OPAQUE

// System cursors
#define GFX_SYSTEM_CURSOR_ARROW SDL_SYSTEM_CURSOR_ARROW
#define GFX_SYSTEM_CURSOR_IBEAM SDL_SYSTEM_CURSOR_IBEAM
#define GFX_SYSTEM_CURSOR_HAND SDL_SYSTEM_CURSOR_HAND

// Blend modes
#define GFX_BLENDMODE_NONE SDL_BLENDMODE_NONE
#define GFX_BLENDMODE_BLEND SDL_BLENDMODE_BLEND
#define GFX_BLENDMODE_ADD SDL_BLENDMODE_ADD
#define GFX_BLENDMODE_MOD SDL_BLENDMODE_MOD

// Keycodes
#define GFXK_BACKSPACE SDLK_BACKSPACE
#define GFXK_DELETE SDLK_DELETE
#define GFXK_UP SDLK_UP
#define GFXK_HOME SDLK_HOME
#define GFXK_DOWN SDLK_DOWN
#define GFXK_END SDLK_END
#define GFXK_LEFT SDLK_LEFT
#define GFXK_RIGHT SDLK_RIGHT

// Letters
#define GFXK_a SDLK_a
#define GFXK_b SDLK_b
#define GFXK_c SDLK_c
#define GFXK_d SDLK_d
#define GFXK_e SDLK_e
#define GFXK_f SDLK_f
#define GFXK_g SDLK_g
#define GFXK_h SDLK_h
#define GFXK_i SDLK_i
#define GFXK_j SDLK_j
#define GFXK_k SDLK_k
#define GFXK_l SDLK_l
#define GFXK_m SDLK_m
#define GFXK_n SDLK_n
#define GFXK_o SDLK_o
#define GFXK_p SDLK_p
#define GFXK_q SDLK_q
#define GFXK_r SDLK_r
#define GFXK_s SDLK_s
#define GFXK_t SDLK_t
#define GFXK_u SDLK_u
#define GFXK_v SDLK_v
#define GFXK_w SDLK_w
#define GFXK_x SDLK_x
#define GFXK_y SDLK_y
#define GFXK_z SDLK_z

// Numbers
#define GFXK_0 SDLK_0
#define GFXK_1 SDLK_1
#define GFXK_2 SDLK_2
#define GFXK_3 SDLK_3
#define GFXK_4 SDLK_4
#define GFXK_5 SDLK_5
#define GFXK_6 SDLK_6
#define GFXK_7 SDLK_7
#define GFXK_8 SDLK_8
#define GFXK_9 SDLK_9

// Symbols
#define GFXK_SPACE SDLK_SPACE
#define GFXK_PLUS SDLK_PLUS
#define GFXK_MINUS SDLK_MINUS
#define GFXK_UNDERSCORE SDLK_UNDERSCORE
#define GFXK_EXCLAIM SDLK_EXCLAIM
#define GFXK_AT SDLK_AT
#define GFXK_HASH SDLK_HASH
#define GFXK_DOLLAR SDLK_DOLLAR
#define GFXK_ASTERISK SDLK_ASTERISK
#define GFXK_QUESTION SDLK_QUESTION
#define GFXK_CARET SDLK_CARET
#define GFXK_LEFTPAREN SDLK_LEFTPAREN
#define GFXK_RIGHTPAREN SDLK_RIGHTPAREN
#define GFXK_AMPERSAND SDLK_AMPERSAND
#define GFXK_PERIOD SDLK_PERIOD
#define GFXK_COMMA SDLK_COMMA
#define GFXK_LESS SDLK_LESS
#define GFXK_GREATER SDLK_GREATER
#define GFXK_SLASH SDLK_SLASH
#define GFXK_BACKSLASH SDLK_BACKSLASH
#define GFXK_COLON SDLK_COLON
#define GFXK_SEMICOLON SDLK_SEMICOLON
#define GFXK_LEFTBRACKET SDLK_LEFTBRACKET
#define GFXK_RIGHTBRACKET SDLK_RIGHTBRACKET
#define GFXK_EQUALS SDLK_EQUALS
#define GFXK_PERCENT SDLK_PERCENT

// Key modifiers
#define GFXMOD_SHIFT KMOD_SHIFT
#define GFXMOD_LSHIFT KMOD_LSHIFT
#define GFXMOD_RSHIFT KMOD_RSHIFT
#define GFXMOD_CAPS KMOD_CAPS

// Lightweight wrappers to avoid direct SDL_* usage in non-Graphics code
#define GFX_DestroyTexture SDL_DestroyTexture
#define GFX_CreateRGBSurface SDL_CreateRGBSurface
#define GFX_FreeSurface SDL_FreeSurface
#define GFX_ConvertSurface SDL_ConvertSurface
#define GFX_BlitScaled SDL_BlitScaled
#define GFX_FillRect SDL_FillRect
#define GFX_SetSurfaceBlendMode SDL_SetSurfaceBlendMode
#define GFX_MapRGBA SDL_MapRGBA
#define GFX_MapRGB SDL_MapRGB
#define GFX_GetRGBA SDL_GetRGBA
#define GFX_GetError SDL_GetError
#define GFX_CreateSystemCursor SDL_CreateSystemCursor
#define GFX_SetCursor SDL_SetCursor

// Image helpers and texture creation
#define GFX_IMG_Load IMG_Load
#define GFX_IMG_GetError IMG_GetError
#define GFX_CreateTextureFromSurface SDL_CreateTextureFromSurface

#else

// Minimal SDL-like stubs so OpenGL path compiles without SDL installed
typedef struct { Uint8 r, g, b, a; } SDL_Color;
typedef struct { int x, y; } SDL_Point;
typedef struct { int x, y, w, h; } SDL_Rect;

typedef struct SDL_Renderer_Opaque SDL_Renderer; // forward opaque
typedef struct SDL_Window_Opaque SDL_Window; // forward opaque
typedef void* SDL_GLContext;
typedef int SDL_Keycode;
typedef int SDL_SystemCursor;
typedef struct SDL_Cursor_Opaque SDL_Cursor;

// Event and related small structs
typedef struct { SDL_Keycode sym; Uint16 mod; } SDL_Keysym;
typedef struct { SDL_Keysym keysym; } SDL_KeyboardEvent;
typedef struct { int x, y; Uint8 button; } SDL_MouseButtonEvent;
typedef struct { int x, y; } SDL_MouseMotionEvent;
typedef struct { int x, y; int direction; } SDL_MouseWheelEvent;
typedef struct SDL_Event {
    int type;
    SDL_KeyboardEvent key;
    SDL_MouseButtonEvent button;
    SDL_MouseMotionEvent motion;
    SDL_MouseWheelEvent wheel;
} SDL_Event;

// Common constants used in code
enum {
    SDL_QUIT = 0x100,
    SDL_KEYDOWN = 0x300,
    SDL_KEYUP,
    SDL_MOUSEMOTION = 0x400,
    SDL_MOUSEBUTTONDOWN,
    SDL_MOUSEBUTTONUP,
    SDL_MOUSEWHEEL
};

// Mouse buttons
#define SDL_BUTTON_LEFT   1
#define SDL_BUTTON_MIDDLE 2
#define SDL_BUTTON_RIGHT  3

// Mouse wheel direction
#define SDL_MOUSEWHEEL_NORMAL 0

// Alpha
#define SDL_ALPHA_OPAQUE 255

// Byte order (assume little-endian on common targets)
#define SDL_BIG_ENDIAN 4321
#define SDL_LIL_ENDIAN 1234
#define SDL_BYTEORDER SDL_LIL_ENDIAN
#define GFX_BYTEORDER SDL_BYTEORDER
#define GFX_BIG_ENDIAN SDL_BIG_ENDIAN

// Key modifiers bitflags
#define KMOD_SHIFT 0x0001
#define KMOD_LSHIFT 0x0002
#define KMOD_RSHIFT 0x0004
#define KMOD_CTRL 0x0008
#define KMOD_ALT 0x0010
#define KMOD_CAPS 0x0020

// Public GFXMOD_* aliases expected by code
#define GFXMOD_SHIFT KMOD_SHIFT
#define GFXMOD_LSHIFT KMOD_LSHIFT
#define GFXMOD_RSHIFT KMOD_RSHIFT
#define GFXMOD_CAPS KMOD_CAPS

// Common keycodes (letters/numbers use ASCII)
#define SDLK_a 'a'
#define SDLK_b 'b'
#define SDLK_c 'c'
#define SDLK_d 'd'
#define SDLK_e 'e'
#define SDLK_f 'f'
#define SDLK_g 'g'
#define SDLK_h 'h'
#define SDLK_i 'i'
#define SDLK_j 'j'
#define SDLK_k 'k'
#define SDLK_l 'l'
#define SDLK_m 'm'
#define SDLK_n 'n'
#define SDLK_o 'o'
#define SDLK_p 'p'
#define SDLK_q 'q'
#define SDLK_r 'r'
#define SDLK_s 's'
#define SDLK_t 't'
#define SDLK_u 'u'
#define SDLK_v 'v'
#define SDLK_w 'w'
#define SDLK_x 'x'
#define SDLK_y 'y'
#define SDLK_z 'z'

#define SDLK_0 '0'
#define SDLK_1 '1'
#define SDLK_2 '2'
#define SDLK_3 '3'
#define SDLK_4 '4'
#define SDLK_5 '5'
#define SDLK_6 '6'
#define SDLK_7 '7'
#define SDLK_8 '8'
#define SDLK_9 '9'

// Non-ASCII keys
#define SDLK_SPACE      0x0100
#define SDLK_TAB        0x0101
#define SDLK_BACKSPACE  0x0102
#define SDLK_DELETE     0x0103
#define SDLK_ESCAPE     0x0104
#define SDLK_UP         0x0105
#define SDLK_DOWN       0x0106
#define SDLK_LEFT       0x0107
#define SDLK_RIGHT      0x0108
#define SDLK_HOME       0x0109
#define SDLK_END        0x010A
#define SDLK_MINUS      0x010B
#define SDLK_EQUALS     0x010C
#define SDLK_LEFTBRACKET 0x010D
#define SDLK_RIGHTBRACKET 0x010E
#define SDLK_BACKSLASH  0x010F
#define SDLK_SEMICOLON  0x0110
#define SDLK_QUOTE      0x0111
#define SDLK_BACKQUOTE  0x0112
#define SDLK_COMMA      0x0113
#define SDLK_PERIOD     0x0114
#define SDLK_SLASH      0x0115

#define SDLK_LCTRL      0x0120
#define SDLK_RCTRL      0x0121
#define SDLK_LALT       0x0122
#define SDLK_RALT       0x0123

// System cursors
enum {
    SDL_SYSTEM_CURSOR_ARROW = 0,
    SDL_SYSTEM_CURSOR_IBEAM = 1,
    SDL_SYSTEM_CURSOR_HAND = 2
};

// Map SDL-like types to Gfx aliases
typedef SDL_Color GfxColor;
typedef SDL_Point GfxPoint;
typedef SDL_Rect GfxRect;
struct GfxTextureOpaque; typedef GfxTextureOpaque GfxTexture;
struct GfxSurfaceOpaque; typedef GfxSurfaceOpaque GfxSurface;
struct TTF_Font_Opaque; typedef TTF_Font_Opaque TTF_Font_Type;
typedef SDL_Cursor GfxCursor;
typedef SDL_Event GfxEvent;
typedef SDL_Keycode GfxKeycode;
typedef SDL_SystemCursor GfxSystemCursor;
struct GfxNativeRendererOpaque; typedef GfxNativeRendererOpaque GfxNativeRenderer;

// Event/type aliases (avoid SDL_ in non-Graphics code)
#define GFX_MOUSEBUTTONDOWN SDL_MOUSEBUTTONDOWN
#define GFX_MOUSEBUTTONUP SDL_MOUSEBUTTONUP
#define GFX_MOUSEMOTION SDL_MOUSEMOTION
#define GFX_MOUSEWHEEL SDL_MOUSEWHEEL
#define GFX_KEYDOWN SDL_KEYDOWN
#define GFX_KEYUP SDL_KEYUP

#define GFX_ALPHA_OPAQUE SDL_ALPHA_OPAQUE

// System cursors
#define GFX_SYSTEM_CURSOR_ARROW SDL_SYSTEM_CURSOR_ARROW
#define GFX_SYSTEM_CURSOR_IBEAM SDL_SYSTEM_CURSOR_IBEAM
#define GFX_SYSTEM_CURSOR_HAND SDL_SYSTEM_CURSOR_HAND

// Blend modes (placeholders for compile-time only)
#define GFX_BLENDMODE_NONE 0
#define GFX_BLENDMODE_BLEND 1
#define GFX_BLENDMODE_ADD 2
#define GFX_BLENDMODE_MOD 3

// Keycodes mapping
#define GFXK_BACKSPACE SDLK_BACKSPACE
#define GFXK_DELETE SDLK_DELETE
#define GFXK_UP SDLK_UP
#define GFXK_HOME SDLK_HOME
#define GFXK_DOWN SDLK_DOWN
#define GFXK_END SDLK_END
#define GFXK_LEFT SDLK_LEFT
#define GFXK_RIGHT SDLK_RIGHT

#define GFXK_a SDLK_a
#define GFXK_b SDLK_b
#define GFXK_c SDLK_c
#define GFXK_d SDLK_d
#define GFXK_e SDLK_e
#define GFXK_f SDLK_f
#define GFXK_g SDLK_g
#define GFXK_h SDLK_h
#define GFXK_i SDLK_i
#define GFXK_j SDLK_j
#define GFXK_k SDLK_k
#define GFXK_l SDLK_l
#define GFXK_m SDLK_m
#define GFXK_n SDLK_n
#define GFXK_o SDLK_o
#define GFXK_p SDLK_p
#define GFXK_q SDLK_q
#define GFXK_r SDLK_r
#define GFXK_s SDLK_s
#define GFXK_t SDLK_t
#define GFXK_u SDLK_u
#define GFXK_v SDLK_v
#define GFXK_w SDLK_w
#define GFXK_x SDLK_x
#define GFXK_y SDLK_y
#define GFXK_z SDLK_z

#define GFXK_0 SDLK_0
#define GFXK_1 SDLK_1
#define GFXK_2 SDLK_2
#define GFXK_3 SDLK_3
#define GFXK_4 SDLK_4
#define GFXK_5 SDLK_5
#define GFXK_6 SDLK_6
#define GFXK_7 SDLK_7
#define GFXK_8 SDLK_8
#define GFXK_9 SDLK_9

#define GFXK_SPACE SDLK_SPACE
#define GFXK_PLUS '+'
#define GFXK_MINUS SDLK_MINUS
#define GFXK_UNDERSCORE '_'
#define GFXK_EXCLAIM '!'
#define GFXK_AT '@'
#define GFXK_HASH '#'
#define GFXK_DOLLAR '$'
#define GFXK_ASTERISK '*'
#define GFXK_QUESTION '?'
#define GFXK_CARET '^'
#define GFXK_LEFTPAREN '('
#define GFXK_RIGHTPAREN ')'
#define GFXK_AMPERSAND '&'
#define GFXK_PERIOD SDLK_PERIOD
#define GFXK_COMMA SDLK_COMMA
#define GFXK_LESS '<'
#define GFXK_GREATER '>'
#define GFXK_SLASH SDLK_SLASH
#define GFXK_BACKSLASH SDLK_BACKSLASH
#define GFXK_COLON ':'
#define GFXK_SEMICOLON SDLK_SEMICOLON
#define GFXK_LEFTBRACKET SDLK_LEFTBRACKET
#define GFXK_RIGHTBRACKET SDLK_RIGHTBRACKET
#define GFXK_EQUALS SDLK_EQUALS
#define GFXK_PERCENT '%'

// Lightweight wrappers no-ops when SDL is absent
#define GFX_DestroyTexture(...)
#define GFX_CreateRGBSurface(...) (NULL)
#define GFX_FreeSurface(...)
#define GFX_ConvertSurface(...) (NULL)
#define GFX_BlitScaled(...) (0)
#define GFX_FillRect(...)
#define GFX_SetSurfaceBlendMode(...)
#define GFX_MapRGBA(...) (0)
#define GFX_MapRGB(...) (0)
#define GFX_GetRGBA(...)
#define GFX_GetError() ("SDL2 not enabled")
#define GFX_CreateSystemCursor(...) (NULL)
#define GFX_SetCursor(...)
#define GFX_IMG_Load(...) (NULL)
#define GFX_IMG_GetError() ("SDL2_image not enabled")
#define GFX_CreateTextureFromSurface(...) (NULL)

#endif // GFX_HAVE_SDL2

#endif

