// Confidential, unpublished property of Robert Carneiro

// The access and distribution of this material is limited solely to
// authorized personnel.  The use, disclosure, reproduction,
// modification, transfer, or transmittal of this work for any purpose
// in any form or by any means without the written permission of
// Robert Carneiro is strictly prohibited.
#ifndef _ML_TRAIN
#define _ML_TRAIN

#include "../crt0.h"
#include "../main.h"
#include "Backend/Database/GList.h"
#include "Backend/Database/GTable.h"
#include "Backend/Database/ServiceData.h"
#include "Backend/Machine Learning/DataObjects/ImageInput.h"
#include "Backend/Machine Learning/DataObjects/NumberInput.h"
#include "Backend/Machine Learning/Networks/metanetwork.h"
#include "Backend/Machine Learning/Networks/network.h"
#include "Backend/Machine Learning/State/Terminator.h"
#include "Backend/Machine Learning/Structure/hiddenlayerinfo.h"
#include "Backend/Machine Learning/Structure/inputlayerinfo.h"
#include "Backend/Machine Learning/Structure/nninfo.h"
#include "Backend/Machine Learning/Structure/outputlayerinfo.h"
#include "Backend/Machine Learning/main.h"
#include "Backend/Networking/service.h"
#include "Frontend/GUI/RUMsgBox.h"
#include "Frontend/Graphics/graphics.h"
#include <string>

class NNInfo;

class ML_Train : public GNet::Service
{
private:
	GNet::GServer* serverInstance;
	glades::NNetwork cNetwork;

public:
	ML_Train()
	{
		serverInstance = NULL;
	}

	ML_Train(GNet::GServer* newInstance)
	{
		serverInstance = newInstance;
	}

	~ML_Train()
	{
		serverInstance = NULL; // Not ours to delete
	}

	shmea::ServiceData* execute(const shmea::ServiceData* data)
	{
		class GNet::Connection* destination = data->getConnection();

		if (data->getType() != shmea::ServiceData::TYPE_LIST)
			return NULL;

		shmea::GList cList = data->getList();

		if ((cList.size() == 1) && (cList.getString(0) == "KILL"))
		{
			if (!cNetwork.getRunning())
				return NULL;

			cNetwork.stop();
			printf("\n!!---KILLING NET---!!\n");
		}

		if (cList.size() < 3)
			return NULL;

		shmea::GString modelName = cList.getString(0);
		shmea::GString inputFName = cList.getString(1);
		int inputType = cList.getInt(2);

		// Optional modern config payload:
		//  [3]=netType (int)
		//  [4]=lrScheduleType (int: 0 none, 1 step, 2 exp, 3 cosine)
		//  [5]=stepSize (int)
		//  [6]=gamma (float)
		//  [7]=tMax (int)
		//  [8]=minMult (float)
		//  [9]=globalGradClipNorm (float)
		//  [10]=perElementGradClip (float)
		//  [11]=tbpttWindowOverride (int)
		//
		// Backward compatibility:
		// - If [3] is a string, treat it as legacy weights filename (nn-state) and ignore modern fields.
		int netTypeOverride = -1;
		int schedType = 0;
		int stepSize = 0;
		float gamma = 1.0f;
		int tMax = 0;
		float minMult = 0.0f;
		float clipNorm = 0.0f;
		float perElemClip = 10.0f;
		int tbpttOverride = 0;
		// Legacy (pre-unified) weights filename (database/nn-state/<file>).
		// This service no longer loads these directly because NNetwork's internal graph state
		// is private; unified model packages should be used instead.
		shmea::GString legacyWeightsName = " ";
		bool hasLegacyWeights = false;

		int idx = 3;
		if (cList.size() > idx)
		{
			const shmea::GType t = cList[idx];
			if (t.getType() == shmea::GType::STRING_TYPE)
			{
				legacyWeightsName = cList.getString(idx);
				hasLegacyWeights = (legacyWeightsName != " ");
			}
			else
			{
				netTypeOverride = cList.getInt(idx);
				++idx;
				if (cList.size() > idx) { schedType = cList.getInt(idx); ++idx; }
				if (cList.size() > idx) { stepSize = cList.getInt(idx); ++idx; }
				if (cList.size() > idx) { gamma = cList.getFloat(idx); ++idx; }
				if (cList.size() > idx) { tMax = cList.getInt(idx); ++idx; }
				if (cList.size() > idx) { minMult = cList.getFloat(idx); ++idx; }
				if (cList.size() > idx) { clipNorm = cList.getFloat(idx); ++idx; }
				if (cList.size() > idx) { perElemClip = cList.getFloat(idx); ++idx; }
				if (cList.size() > idx) { tbpttOverride = cList.getInt(idx); ++idx; }
			}
		}
		/*int64_t maxTimeStamp = cList.getLong(3);
		int64_t maxEpoch = cList.getLong(4);
		float maxAccuracy = cList.getFloat(5);*/
		// int64_t trainPct = cList.getLong(4), testPct = cList.getLong(5), validationPct =
		// cList.getLong(6);

		// Modify the paths to properly load the data later
		glades::DataInput* di = NULL;
		if (inputType == glades::DataInput::CSV)
		{
			inputFName = "datasets/" + inputFName;
			di = new glades::NumberInput();
		}
		else if (inputType == glades::DataInput::IMAGE)
		{
			// inputFName = "datasets/images/" + inputFName + "/";
			di = new glades::ImageInput();
		}
		else if (inputType == glades::DataInput::TEXT)
		{
			// TODO
			return NULL;
		}
		else
			return NULL;

		if (!di)
			return NULL;

		// Load the input data
		di->import(inputFName);

		// Load the model (unified model package preferred; legacy fallback).
		if (cNetwork.getEpochs() == 0)
		{
			const glades::NNetworkStatus st = cNetwork.loadModel(std::string(modelName.c_str()), di, netTypeOverride);
			if (!st.ok())
			{
				if (!cNetwork.load(modelName))
				{
					printf("[NN] Unable to load model \"%s\": %s\n", modelName.c_str(), st.message.c_str());
					return NULL;
				}
			}
		}

		// Apply optional overrides (modern features).
		if (netTypeOverride >= 0 && netTypeOverride <= 3)
		{
			// netType is a constructor-time trait, but loadModel(netTypeOverride) handles override.
			// If we're on legacy load(), best-effort set cNetwork.netType via a rebuild path inside training.
		}
		if (schedType == 1)
			cNetwork.setLearningRateScheduleStep(stepSize, gamma);
		else if (schedType == 2)
			cNetwork.setLearningRateScheduleExp(gamma);
		else if (schedType == 3)
			cNetwork.setLearningRateScheduleCosine(tMax, minMult);
		else
			cNetwork.setLearningRateScheduleNone();
		cNetwork.setGlobalGradClipNorm(clipNorm);
		cNetwork.setPerElementGradClip(perElemClip);
		cNetwork.getTrainingConfigMutable().tbpttWindowOverride = tbpttOverride;

		// Legacy weights load (pre-unified persistence).
		// Intentionally not supported here anymore; these should be migrated into model packages.
		if (hasLegacyWeights)
			printf("[NN] note: legacy nn-state weights \"%s\" ignored; use unified model packages instead\n",
			       legacyWeightsName.c_str());

		// Termination Conditions
		/*cNetwork.terminator.setTimestamp(maxTimeStamp);
		cNetwork.terminator.setEpoch(maxEpoch);
		cNetwork.terminator.setAccuracy(maxAccuracy);*/

		// Run the training and retrieve a metanetwork
		glades::MetaNetwork* newTrainNet =
			glades::train(&cNetwork, di, serverInstance, destination);
		cNetwork.setMustdBuildMeat(true);

		return NULL;
	}

	GNet::Service* MakeService(GNet::GServer* newInstance) const
	{
		return new ML_Train(newInstance);
	}

	shmea::GString getName() const
	{
		return "ML_Train";
	}
};

#endif
