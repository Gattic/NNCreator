// Copyright 2020 Robert Carneiro, Derek Meer, Matthew Tabak, Eric Lujan
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

#ifndef _RUGENERALGRAPHABLE_H
#define _RUGENERALGRAPHABLE_H

#include "Graphable.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include <map>

// Replicate a vtable with virtual functions.
class GeneralGraphable
{
	class GraphableConcept
	{
	public:
		virtual ~GraphableConcept()
		{
			//
		}

		virtual bool isVisible() const = 0;
		virtual unsigned int size() const = 0;
		virtual unsigned int normalizedSize() const = 0;
		virtual void setVisible(bool newVisible) = 0;
		virtual void setColor(SDL_Color newColor) = 0;
		virtual void computeAxisRanges(bool additionOptimization = false) = 0;
		virtual void updateBackground(gfxpp* cGfx) = 0;
	};

	template< typename T>
	class GraphableModel : public GraphableConcept
	{
	protected:

		Graphable<T>* g;

	public:

		GraphableModel()
		{
			//
		}

		GraphableModel(RUGraph* newGraph, SDL_Color newColor)
		{
			g = new Graphable<T>(newGraph, newColor);
		}

		void set(const std::vector<T*>& newPoints)
		{
			g->set(newPoints);
		}
		
		// This function is recommended for TimeSeries optimizations.
		void add(const T* newPoint, bool recompute)
		{
			g->add(newPoint, recompute);
		}

		bool isVisible() const
		{
			return g->isVisible();
		}

		unsigned int size() const
		{
			return g->size();
		}

		unsigned int normalizedSize() const
		{
			return g->normalizedSize();
		}

		void setVisible(bool newVisible)
		{
			g->setVisible(newVisible);
		}

		void setColor(SDL_Color newColor)
		{
			g->setColor(newColor);
		}

		void computeAxisRanges(bool additionOptimization)
		{
			g->computeAxisRanges(additionOptimization);
		}

		void updateBackground(gfxpp* cGfx)
		{
			g->updateBackground(cGfx);
		}

		Graphable<T>* get(T t)
		{
			return g;
		}

		virtual ~GraphableModel()
		{
			//
		}
	};

	GraphableConcept* object;

public:

	GeneralGraphable()
	{
		object = NULL;
	}

	template< typename T>
	GeneralGraphable(RUGraph* newGraph, SDL_Color newColor, T t) // Explicit templates on consctuctors do not work
	{
		object = new GraphableModel<T>(newGraph, newColor);
	}

	template< typename T>
	void add(const T* newPoint, bool recompute)
	{
		if(object)
			((GraphableModel<T>*)object)->add(newPoint, recompute);
	}

	template< typename T>
	void set(const std::vector<T*>& newPoints)
	{
		if(object)
			((GraphableModel<T>*)object)->set(newPoints);
	}

	bool isVisible() const
	{
		if(object)
			return object->isVisible();
		return false;
	}

	unsigned int size() const
	{
		if(object)
			return object->size();
		return 0;
	}

	unsigned int normalizedSize() const
	{
		if(object)
			return object->normalizedSize();
		return 0;
	}

	void setVisible(bool newVisible)
	{
		if(object)
			object->setVisible(newVisible);
	}

	void setColor(SDL_Color newColor)
	{
		if(object)
			object->setColor(newColor);
	}

	void computeAxisRanges(bool additionOptimization = false)
	{
		if(object)
			object->computeAxisRanges(additionOptimization);
	}

	void updateBackground(gfxpp* cGfx)
	{
		if(object)
			object->updateBackground(cGfx);
	}

	template< typename T>
	Graphable<T>* get(T t)
	{
		if(object)
			return ((GraphableModel<T>*)object)->get(t);
		return NULL;
	}

	virtual ~GeneralGraphable()
	{
		//
	}
};

#endif
