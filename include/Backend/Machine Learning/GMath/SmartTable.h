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
#ifndef _SMARTTABLE
#define _SMARTTABLE

#include "Backend/Database/GTable.h"
#include <vector>

namespace shmea {
class GTable;
};

namespace glades {

class OHE;

class SmartTable : public shmea::GTable
{
private:
    shmea::GTable OHEMaps;
    std::vector<bool> featureIsCategorical;

public:

    std::map<std::string, double> classCount;

    // standardization flags
    static const int MINMAX = 0;
    static const int ZSCORE = 1;

    SmartTable() : shmea::GTable() { }

    SmartTable(char newDelimiter) : shmea::GTable(newDelimiter) { }

    SmartTable(char newDelimiter, const std::vector<shmea::GString>& newHeaders)
	: shmea::GTable(newDelimiter, newHeaders) { }

    SmartTable(const shmea::GString& fname, char newDelimiter, int importFlag)
	: shmea::GTable(fname, newDelimiter, importFlag) { }

    SmartTable(const SmartTable& gtable2) : shmea::GTable(gtable2) { }

    virtual ~SmartTable();

    virtual void standardize(int = 0);
    virtual float unstandardize(float) const;

};
};

#endif
