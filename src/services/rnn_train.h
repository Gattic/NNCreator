// Copyright 2020 Robert Carneiro, Derek Meer, Matthew Tabak, Eric Lujan
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and
// associated documentation files (the "Software"), to deal in the Software
// without restriction,
// including without limitation the rights to use, copy, modify, merge, publish,
// distribute,
// sublicense, and/or sell copies of the Software, and to permit persons to whom
// the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT
// NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
// PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
// FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
// OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#ifndef _RNN_TRAIN
#define _RNN_TRAIN

#include "../crt0.h"
#include "../main.h"
#include "Backend/Database/GList.h"
#include "Backend/Database/gtable.h"
#include "Backend/Machine Learning/RNN.h"
#include "Backend/Machine Learning/State/Terminator.h"
#include "Backend/Machine Learning/Structure/hiddenlayerinfo.h"
#include "Backend/Machine Learning/Structure/inputlayerinfo.h"
#include "Backend/Machine Learning/Structure/nninfo.h"
#include "Backend/Machine Learning/Structure/outputlayerinfo.h"
#include "Backend/Machine Learning/main.h"
#include "Backend/Machine Learning/metanetwork.h"
#include "Backend/Machine Learning/network.h"
#include "Backend/Networking/service.h"
#include "Backend/Networking/socket.h"
#include "Frontend/GUI/RUMsgBox.h"
#include "Frontend/Graphics/graphics.h"

class NNInfo;

class RNN_Train : public Service
{
public:
	GList execute(class Instance* cInstance, const GList& data)
	{
		GList retList;
		if (data.size() < 3)
			return retList;

		std::string netName = data.getString(0);
		std::string testFName = data.getString(1);
		int importType = data.getInt(2);
		int64_t maxTimeStamp = data.getLong(3);
		int64_t maxEpoch = data.getLong(4);
		float maxAccuracy = data.getFloat(5);
		// int64_t trainPct = data.getLong(4), testPct = data.getLong(5),
		// validationPct =
		// data.getLong(6);

		// load the neural network
		RNN* cNetwork = GQL::getRNN(netName);

		// Termination Conditions
		Terminator* Arnold = new Terminator();
		Arnold->setTimestamp(maxTimeStamp);
		Arnold->setEpoch(maxEpoch);
		Arnold->setAccuracy(maxAccuracy);

		// Run the training and retrieve a metanetwork
		GTable inputTable(testFName, ',', importType);
		MetaNetwork* newTrainNet = GQL::train(cNetwork, inputTable, Arnold);

		return retList;
	}
};

#endif
