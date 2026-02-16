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
#ifndef _ML_TEST
#define _ML_TEST

#include "../crt0.h"
#include "../main.h"
#include "Backend/Database/GList.h"
#include "Backend/Database/GTable.h"
#include "Backend/Database/ServiceData.h"
#include "Backend/Machine Learning/DataObjects/ImageInput.h"
#include "Backend/Machine Learning/DataObjects/NumberInput.h"
#include "Backend/Machine Learning/DataObjects/TokenInput.h"
#include "Backend/Machine Learning/Networks/metanetwork.h"
#include "Backend/Machine Learning/Networks/network.h"
#include "Backend/Machine Learning/Networks/training_callbacks.h"
#include "Backend/Machine Learning/State/Terminator.h"
#include "Backend/Machine Learning/Structure/hiddenlayerinfo.h"
#include "Backend/Machine Learning/Structure/inputlayerinfo.h"
#include "Backend/Machine Learning/Structure/nninfo.h"
#include "Backend/Machine Learning/Structure/outputlayerinfo.h"
#include "Backend/Machine Learning/main.h"
#include "Backend/Networking/service.h"
#include "Frontend/GItems/GPanel.h"
#include "ml_train.h"
#include <string>

class ML_Test : public GNet::Service
{
private:
	GNet::GServer* serverInstance;
	GPanel* cPanel;
	bool* killFlag;
	glades::NNetwork cNetwork;

public:
	ML_Test()
	{
		serverInstance = NULL;
		cPanel = NULL;
		killFlag = NULL;
	}

	ML_Test(GNet::GServer* newInstance)
	{
		serverInstance = newInstance;
		cPanel = NULL;
		killFlag = NULL;
	}

	ML_Test(GNet::GServer* newInstance, GPanel* newPanel, bool* kf = NULL)
	{
		serverInstance = newInstance;
		cPanel = newPanel;
		killFlag = kf;
	}

	~ML_Test()
	{
		serverInstance = NULL;
		cPanel = NULL;
		killFlag = NULL;
	}

	shmea::ServiceData* execute(const shmea::ServiceData* data)
	{
		class GNet::Connection* destination = data->getConnection();

		if (data->getType() != shmea::ServiceData::TYPE_LIST)
			return NULL;

		shmea::GList cList = data->getList();
		if (cList.size() < 3)
			return NULL;

		shmea::GString modelName = cList.getString(0);
		shmea::GString inputFName = cList.getString(1);
		int inputType = cList.getInt(2);

		// Create the appropriate DataInput
		glades::DataInput* di = NULL;
		if (inputType == glades::DataInput::CSV)
		{
			inputFName = "datasets/" + inputFName;
			di = new glades::NumberInput();
		}
		else if (inputType == glades::DataInput::IMAGE)
		{
			di = new glades::ImageInput();
		}
		else if (inputType == glades::DataInput::TEXT)
		{
			inputFName = "datasets/" + inputFName;
			di = new glades::TokenInput();
		}
		else
			return NULL;

		if (!di)
			return NULL;

		// Load the input data
		di->import(inputFName);

		// Load the model
		if (cNetwork.getEpochs() == 0)
		{
			const glades::NNetworkStatus st = cNetwork.loadModel(std::string(modelName.c_str()), di);
			if (!st.ok())
			{
				printf("[NN] Unable to load model \"%s\": %s\n", modelName.c_str(), st.message.c_str());
				delete di;
				return NULL;
			}
		}

		// Run testing with direct panel callbacks that bypass the socket/service
		// framework, same as ML_Train.
		DirectPanelCallbacks panelCb(cPanel, killFlag);
		glades::MetaNetwork* result = glades::test(&cNetwork, di,
			cPanel ? static_cast<glades::ITrainingCallbacks*>(&panelCb) : static_cast<glades::ITrainingCallbacks*>(NULL),
			serverInstance, destination);
		delete result;
		delete di;

		return NULL;
	}

	GNet::Service* MakeService(GNet::GServer* newInstance) const
	{
		return new ML_Test(newInstance, cPanel, killFlag);
	}

	shmea::GString getName() const
	{
		return "ML_Test";
	}
};

#endif
