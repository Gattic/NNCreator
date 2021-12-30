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

#ifndef _RUGENERALLISTENER
#define _RUGENERALLISTENER

#include "Backend/Database/GString.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include <map>

//TODO:
// MAKE GITEMS = EVENTS (use operators)

// Replicate a vtable with virtual functions.
// Could also do this with function pointers
//class GeneralListener
class GeneralListener
{
	class ListenerConcept
	{
	public:
		virtual ~ListenerConcept()
		{
			//
		}

		virtual void call()=0;
		virtual void call(int scrollType)=0;
		virtual void call(int x, int y)=0;
		virtual void call(const shmea::GString& str, int x, int y)=0;
		virtual void call(const shmea::GString& str)=0;
	};

	template< typename T>
	class ListenerModel : public ListenerConcept
	{
	protected:

		T* cbPanel;
		void (T::*listener0)() = NULL;
		void (T::*listener1)(int) = NULL;
		void (T::*listener2)(int, int) = NULL;
		void (T::*listener3)(const shmea::GString&, int, int) = NULL;
		void (T::*listener4)(const shmea::GString&) = NULL;

	public:
		ListenerModel(T* t, void (T::*newListener)()) : cbPanel(t), listener0(newListener)
		{
			//
		}

		ListenerModel(T* t, void (T::*newListener)(int)) : cbPanel(t), listener1(newListener)
		{
			//
		}

		ListenerModel(T* t, void (T::*newListener)(int, int)) : cbPanel(t), listener2(newListener)
		{
			//
		}

		ListenerModel(T* t, void (T::*newListener)(const shmea::GString&, int, int)) : cbPanel(t), listener3(newListener)
		{
			//
		}

		ListenerModel(T* t, void (T::*newListener)(const shmea::GString&)) : cbPanel(t), listener4(newListener)
		{
			//
		}

		void call()
		{
			if(!listener0)
				return;

			(cbPanel->*listener0)();
		}

		void call(int scrollType)
		{
			if(!listener1)
				return;

			(cbPanel->*listener1)(scrollType);
		}

		void call(int x, int y)
		{
			if(!listener2)
				return;

			(cbPanel->*listener2)(x, y);
		}

		void call(const shmea::GString& str, int x, int y)
		{
			if(!listener3)
				return;

			(cbPanel->*listener3)(str, x, y);
		}

		void call(const shmea::GString& str)
		{
			if(!listener4)
				return;

			(cbPanel->*listener4)(str);
		}

		virtual ~ListenerModel()
		{
			//
		}
	};

	ListenerConcept* object;

public:

	GeneralListener()
	{
		object = NULL;
	}

	template< typename T>
	GeneralListener(T* callbackPanel, void (T::*newListener)())
	{
		object = new ListenerModel<T>(callbackPanel, newListener);
	}

	template< typename T>
	GeneralListener(T* callbackPanel, void (T::*newListener)(int))
	{
		object = new ListenerModel<T>(callbackPanel, newListener);
	}

	template< typename T>
	GeneralListener(T* callbackPanel, void (T::*newListener)(int, int))
	{
		object = new ListenerModel<T>(callbackPanel, newListener);
	}

	template< typename T>
	GeneralListener(T* callbackPanel, void (T::*newListener)(const shmea::GString&, int, int))
	{
		object = new ListenerModel<T>(callbackPanel, newListener);
	}

	template< typename T>
	GeneralListener(T* callbackPanel, void (T::*newListener)(const shmea::GString&))
	{
		object = new ListenerModel<T>(callbackPanel, newListener);
	}

	void call()
	{
		if(object)
			object->call();
	}

	void call(int scrollType)
	{
		if(object)
			object->call(scrollType);
	}

	void call(int x, int y)
	{
		if(object)
			object->call(x, y);
	}

	void call(const shmea::GString& str, int x, int y)
	{
		if(object)
			object->call(str, x, y);
	}

	void call(const shmea::GString& str)
	{
		if(object)
			object->call(str);
	}

	virtual ~GeneralListener()
	{
		//
	}
};

void lExample();

#endif
