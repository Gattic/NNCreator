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
#ifndef _RUNNCREATORPANEL
#define _RUNNCREATORPANEL

#include "Backend/Machine Learning/DataObjects/ImageInput.h"
#include "Backend/Machine Learning/glades.h"
#include "Frontend/GItems/GPanel.h"
#include <map>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <time.h>
#include <vector>

class Point2;
class GLinearLayout;
class RUImageComponent;
class RULabel;
class RUTextbox;
class RUCheckbox;
class RUButton;
class RUDropdown;
class RUGraph;
class RUTable;
class RUProgressBar;
class RUTabContainer;
class PlotType;
class DrawNeuralNet;

namespace shmea {
class GTable;
class GList;
}; // namespace shmea

namespace GNet {
class GServer;
};

namespace glades {
class NNInfo;
};

class NNCreatorPanel : public GPanel
{
protected:
	virtual void updateFromQ(const shmea::ServiceData*);
	virtual void onStart();

	GNet::GServer* serverInstance;
	glades::NNInfo* formInfo;
	glades::ImageInput ii;
	int currentHiddenLayerIndex;
	unsigned int netCount;
	bool keepGraping;
	unsigned int trainingRowIndex;
	unsigned int testingRowIndex;
	int prevImageFlag;

	int64_t parsePct(const shmea::GType&);

	void buildPanel();

	DrawNeuralNet* nn;

	RUGraph* lcGraph;
	RUImageComponent* outputImage;
	RUGraph* rocCurveGraph;
	RUTable* cMatrixTable;

	RUGraph* neuralNetGraph;

	RULabel* lblSettings;

	RULabel* lblEpochs;
	RULabel* lblAccuracy;

	RULabel* lblNeuralNet;
	RUDropdown* ddNeuralNet;

	RULabel* lblNetName;
	RUTextbox* tbNetName;

	RUButton* btnSave;
	RUButton* btnDelete;

	RUTabContainer* layerTabs;
	GLinearLayout* inputOverallLayout;
	GLinearLayout* hiddenOverallLayout;
	GLinearLayout* outputOverallLayout;

	RULabel* lblLayerSize;

	RULabel* lblHiddenLayerCount;
	RUTextbox* tbHiddenLayerCount;

	RULabel* lblEditHiddenLayer;
	// RULabel* lblIndexToEdit;
	RUDropdown* ddIndexToEdit;

	RULabel* lblHiddenLayerSize;
	RUTextbox* tbHiddenLayerSize;

	RULabel* lblLearningRate;
	RUTextbox* tbLearningRate;

	RULabel* lblWeightDecay1;
	RUTextbox* tbWeightDecay1;

	RULabel* lblWeightDecay2;
	RUTextbox* tbWeightDecay2;

	RULabel* lblMomentumFactor;
	RUTextbox* tbMomentumFactor;

	RULabel* lblPHidden;
	RUTextbox* tbPHidden;

	RULabel* lblActivationFunctions;
	RUDropdown* ddActivationFunctions;

	RULabel* lblActivationParam;
	RUTextbox* tbActivationParam;

	RULabel* lblEditInputLayer;

	RUTextbox* tbBatchSize;

	RULabel* lblinputLR;
	RUTextbox* tbinputLR;

	RULabel* lblinputWD1;
	RUTextbox* tbinputWD1;

	RULabel* lblinputWD2;
	RUTextbox* tbinputWD2;

	RULabel* lblinputMF;
	RUTextbox* tbinputMF;

	RULabel* lblinputDropout;
	RUTextbox* tbinputDropout;

	RULabel* lblinputAF;
	RUDropdown* ddinputAF;

	RULabel* lblinputAP;
	RUTextbox* tbinputAP;

	RUTabContainer* previewTabs;
	RUTable* previewTable;
	GLinearLayout* previewImageLayout;
	RUImageComponent* previewImage;

	RULabel* lblEditOutputLayer;

	RULabel* lblOutputType;
	RUDropdown* ddOutputType;

	RULabel* lblOutputLayerSize;
	RUTextbox* tbOutputLayerSize;

	RUTextbox* tbCopyDestination;

	RUButton* sendButton;

	RUDropdown* ddDatasets;
	RUDropdown* ddDataType;

	RUCheckbox* chkCrossVal;
	RULabel* lblttv;
	RUTextbox* tbTrainPct;
	RUTextbox* tbTestPct;
	RUTextbox* tbValidationPct;

public:
	pthread_mutex_t* lcMutex;
	pthread_mutex_t* rocMutex;

	NNCreatorPanel(const shmea::GString&, int, int);
	NNCreatorPanel(GNet::GServer*, const shmea::GString&, int, int);
	virtual ~NNCreatorPanel();

	void loadDDNN();
	void populateIndexToEdit(int = 0);
	void populateInputLayerForm();
	void populateHLayerForm();
	void syncFormVar();
	void loadNNet(glades::NNInfo*);
	void PlotLearningCurve(float, float);
	void PlotROCCurve(float, float);
	void updateConfMatrixTable(const shmea::GTable&);

	void loadDatasets();

	void clickedSave(const shmea::GString&, int, int);
	void clickedEditSwitch(const shmea::GString&, int, int);
	void clickedDSTypeSwitch(int);
	void clickedRun(const shmea::GString&, int, int);
	void clickedCopy(const shmea::GString&, int, int);
	void clickedRemove(const shmea::GString&, int, int);
	void tbHLLoseFocus();
	void clickedLoad(const shmea::GString&, int, int);
	void checkedCV(const shmea::GString&, int, int);
	void clickedKill(const shmea::GString&, int, int);
	void clickedContinue(const shmea::GString&, int, int);
	void clickedDelete(const shmea::GString&, int, int);
	void clickedPreviewTrain(const shmea::GString&, int, int);
	void clickedPreviewTest(const shmea::GString&, int, int);
	void clickedPrevious(const shmea::GString&, int, int);
	void clickedNext(const shmea::GString&, int, int);
	void nnSelectorChanged(int);
	void resetSim();
};

#endif
