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
#include "version.h"

Version::Version(const std::string newVersion)
{
	version = newVersion;
}

std::string Version::getString() const
{
	return version;
}

bool Version::operator==(const std::string& clientVersion) const
{
	return (strcmp(getString().c_str(), clientVersion.c_str()) == 0);
}

bool Version::operator==(const Version& Version2) const
{
	return (strcmp(getString().c_str(), Version2.getString().c_str()) == 0);
}

bool Version::operator!=(const std::string& clientVersion) const
{
	return !(*this == clientVersion);
}

bool Version::operator!=(const Version& Version2) const
{
	return !(*this == Version2);
}

const std::string Version::header() const
{
	std::string vString = "";
	vString += "+-------------------------+\n";
	vString += "| NNCreator               |\n";
	vString += "| Version " + getString() + "            |\n";
	vString += "| Author Robert Carneiro  |\n";
	vString += "+-------------------------+\n";
	return vString;
}
