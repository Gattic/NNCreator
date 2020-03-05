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
#include "nncreator.h"
#include "Backend/Database/GList.h"
#include "Backend/Database/gtable.h"
#include "Backend/Database/gtype.h"
#include "Backend/Database/saveitem.h"
#include "Backend/Database/savelist.h"
#include "Backend/Machine Learning/GMath/gmath.h"
#include "Backend/Machine Learning/Structure/hiddenlayerinfo.h"
#include "Backend/Machine Learning/Structure/inputlayerinfo.h"
#include "Backend/Machine Learning/Structure/nninfo.h"
#include "Backend/Machine Learning/Structure/outputlayerinfo.h"
#include "Backend/Machine Learning/glades.h"
#include "Backend/Machine Learning/network.h"
#include "Backend/Networking/instance.h"
#include "Backend/Networking/main.h"
#include "Backend/Networking/service.h"
#include "Frontend/GItems/GItem.h"
#include "Frontend/GLayouts/GLinearLayout.h"
#include "Frontend/GUI/RUCheckbox.h"
#include "Frontend/GUI/RUDropdown.h"
#include "Frontend/GUI/RUGraph/RUGraph.h"
#include "Frontend/GUI/RUImageComponent.h"
#include "Frontend/GUI/RUMsgBox.h"
#include "Frontend/GUI/RUTabContainer.h"
#include "Frontend/GUI/RUTable.h"
#include "Frontend/GUI/Text/RUButton.h"
#include "Frontend/GUI/Text/RULabel.h"
#include "Frontend/GUI/Text/RUTextbox.h"
#include "Frontend/Graphics/graphics.h"
#include "crt0.h"
#include "main.h"

// using namespace shmea;
using namespace glades;

/*!
 * @brief NNCreatorPanel constructor
 * @details builds the NNCreator panel
 * @param name name of the panel
 * @param width width of the panel
 * @param height height of the panel
 */
NNCreatorPanel::NNCreatorPanel(const std::string& name, int width, int height)
	: GPanel(name, width, height)
{
	currentHiddenLayerIndex = 0;
	InputLayerInfo* newInputLayer = new InputLayerInfo(0.0f, 1);
	std::vector<HiddenLayerInfo*> newHiddenLayers;
	newHiddenLayers.push_back(new HiddenLayerInfo(2, 0.01f, 0.0f, 0.0f, 0.0f, 0, 0.0f));
	OutputLayerInfo* newOutputLayer = new OutputLayerInfo(1, OutputLayerInfo::REGRESSION);
	formInfo = new glades::NNInfo("", newInputLayer, newHiddenLayers, newOutputLayer);

	// Stores all the graphs
	GLinearLayout* graphsLayout = new GLinearLayout("graphsLayout");
	graphsLayout->setX(15);
	graphsLayout->setY(60);
	graphsLayout->setPadding(5);
	graphsLayout->setOrientation(GLinearLayout::VERTICAL);
	addSubItem(graphsLayout);

	// Top Row of graphs
	GLinearLayout* topGraphsLayout = new GLinearLayout("topGraphsLayout");
	topGraphsLayout->setPadding(5);
	topGraphsLayout->setOrientation(GLinearLayout::HORIZONTAL);
	graphsLayout->addSubItem(topGraphsLayout);

	// Learning Curve Graph and Label
	GLinearLayout* lcGraphLayout = new GLinearLayout("lcGraphLayout");
	lcGraphLayout->setPadding(5);
	lcGraphLayout->setOrientation(GLinearLayout::VERTICAL);
	topGraphsLayout->addSubItem(lcGraphLayout);

	// Learning Curve Label
	RULabel* lblGraphLC = new RULabel();
	lblGraphLC->setWidth(300);
	lblGraphLC->setHeight(25);
	lblGraphLC->setText("Learning Curve (Epoch, Error)");
	lblGraphLC->setName("lblGraphLC");
	lcGraphLayout->addSubItem(lblGraphLC);

	// Learning curve graph
	lcGraph =
		new RUGraph(Graphics::getWidth() / 4, Graphics::getHeight() / 4, RUGraph::QUADRANTS_ONE);
	lcGraph->setName("lcGraph");
	lcGraphLayout->addSubItem(lcGraph);

	// ROC Curve Graph and Label
	GLinearLayout* rocGraphLayout = new GLinearLayout("rocGraphLayout");
	rocGraphLayout->setPadding(5);
	rocGraphLayout->setOrientation(GLinearLayout::VERTICAL);
	topGraphsLayout->addSubItem(rocGraphLayout);

	// ROC Curve Label
	RULabel* lblGraphROC = new RULabel();
	lblGraphROC->setWidth(350);
	lblGraphROC->setHeight(25);
	lblGraphROC->setText("ROC Curve (False Pos, True pos)");
	lblGraphROC->setName("lblGraphROC");
	rocGraphLayout->addSubItem(lblGraphROC);

	// ROC Curve Graph
	rocCurveGraph = new RUGraph(Graphics::getWidth() / 4, Graphics::getHeight() / 4,
								RUGraph::QUADRANTS_ONE); // -4 = adjustment to align with table
	rocCurveGraph->setName("rocCurveGraph");
	rocGraphLayout->addSubItem(rocCurveGraph);

	// Bottom row of graphs
	GLinearLayout* bottomGraphsLayout = new GLinearLayout("bottomGraphsLayout");
	bottomGraphsLayout->setPadding(5);
	bottomGraphsLayout->setOrientation(GLinearLayout::HORIZONTAL);
	graphsLayout->addSubItem(bottomGraphsLayout);

	// Dartboard Graph and Label
	GLinearLayout* dartGraphLayout = new GLinearLayout("dartGraphLayout");
	dartGraphLayout->setPadding(5);
	dartGraphLayout->setOrientation(GLinearLayout::VERTICAL);
	bottomGraphsLayout->addSubItem(dartGraphLayout);

	// Dartboard Label
	RULabel* lblGraphDart = new RULabel();
	lblGraphDart->setWidth(310);
	lblGraphDart->setHeight(24);
	lblGraphDart->setText("Dartboard (Expected, Predicted)");
	lblGraphDart->setName("lblGraphDart");
	dartGraphLayout->addSubItem(lblGraphDart);

	// Dartboard graph
	dartboardGraph = new RUGraph(Graphics::getWidth() / 4, Graphics::getHeight() / 4,
								 RUGraph::QUADRANTS_ONE); // -4 = adjustment to align with table
	dartboardGraph->setName("dartboardGraph");
	dartGraphLayout->addSubItem(dartboardGraph);

	// Confusion Table and Label
	GLinearLayout* confTableLayout = new GLinearLayout("confTableLayout");
	confTableLayout->setPadding(5);
	confTableLayout->setOrientation(GLinearLayout::VERTICAL);
	bottomGraphsLayout->addSubItem(confTableLayout);

	// Confusion Matrix Label
	RULabel* lblTableConf = new RULabel();
	lblTableConf->setWidth(300);
	lblTableConf->setHeight(25);
	lblTableConf->setText("            Confusion Matrix");
	lblTableConf->setName("lblTableConf");
	confTableLayout->addSubItem(lblTableConf);

	// Confusion Matrix Table
	cMatrixTable = new RUTable();
	cMatrixTable->setRowsShown(5);
	cMatrixTable->setWidth(Graphics::getWidth() / 4);
	cMatrixTable->setHeight(Graphics::getHeight() / 4);
	cMatrixTable->setName("cMatrixTable");
	confTableLayout->addSubItem(cMatrixTable);

	//============LEFT============

	// Left Side forms
	GLinearLayout* leftSideLayout = new GLinearLayout("leftSideLayout");
	leftSideLayout->setX(15);
	leftSideLayout->setY(5 + bottomGraphsLayout->getY() + bottomGraphsLayout->getHeight());
	leftSideLayout->setPadding(10);
	leftSideLayout->setOrientation(GLinearLayout::VERTICAL);
	addSubItem(leftSideLayout);

	// Neural Network Settings header
	lblSettings = new RULabel();
	lblSettings->setWidth(250);
	lblSettings->setHeight(40);
	lblSettings->setPadding(10);
	lblSettings->setText("Neural Network Settings");
	lblSettings->setName("lblSettings");
	leftSideLayout->addSubItem(lblSettings);

	GLinearLayout* loadNetLayout = new GLinearLayout("loadNetLayout");
	loadNetLayout->setOrientation(GLinearLayout::HORIZONTAL);
	leftSideLayout->addSubItem(loadNetLayout);

	// Neural Net selector label
	lblNeuralNet = new RULabel();
	lblNeuralNet->setWidth(200);
	lblNeuralNet->setHeight(30);
	lblNeuralNet->setText("Neural Network");
	lblNeuralNet->setName("lblNeuralNet");
	loadNetLayout->addSubItem(lblNeuralNet);

	// Neural Net selector
	ddNeuralNet = new RUDropdown();
	ddNeuralNet->setWidth(220);
	ddNeuralNet->setHeight(30);
	ddNeuralNet->setOptionsShown(3);
	ddNeuralNet->setName("ddNeuralNet");
	ddNeuralNet->setOptionChangedListener(&GPanel::GuiCommander15);
	loadNetLayout->addSubItem(ddNeuralNet);

	// Load button
	RUButton* btnLoad = new RUButton();
	btnLoad->setWidth(206);
	btnLoad->setHeight(30);
	btnLoad->setText("      Reload List");
	btnLoad->setMouseDownListener(&GPanel::GuiCommander10);
	btnLoad->setName("btnLoad");
	loadNetLayout->addSubItem(btnLoad);

	GLinearLayout* netNameLayout = new GLinearLayout("netNameLayout");
	netNameLayout->setOrientation(GLinearLayout::HORIZONTAL);
	leftSideLayout->addSubItem(netNameLayout);

	// Network Name label
	lblNetName = new RULabel();
	lblNetName->setX(6);
	lblNetName->setY(6 + lblNeuralNet->getY() + lblNeuralNet->getHeight());
	lblNetName->setWidth(200);
	lblNetName->setHeight(30);
	lblNetName->setText("Network Name");
	lblNetName->setName("lblNetName");
	netNameLayout->addSubItem(lblNetName);

	// Network Name textbox
	tbNetName = new RUTextbox();
	tbNetName->setX(20 + lblNetName->getX() + lblNetName->getWidth());
	tbNetName->setY(lblNetName->getY());
	tbNetName->setWidth(220);
	tbNetName->setHeight(30);
	tbNetName->setText("");
	tbNetName->setName("tbNetName");
	netNameLayout->addSubItem(tbNetName);

	// Save button
	btnSave = new RUButton("green");
	btnSave->setWidth(100);
	btnSave->setHeight(30);
	btnSave->setText("    Save");
	btnSave->setMouseDownListener(&GPanel::GuiCommander1);
	btnSave->setName("btnSave");
	netNameLayout->addSubItem(btnSave);

	// Delete button
	btnDelete = new RUButton("red");
	btnDelete->setWidth(100);
	btnDelete->setHeight(30);
	btnDelete->setText("  Delete");
	btnDelete->setMouseDownListener(&GPanel::GuiCommander14);
	btnDelete->setName("btnDelete");
	netNameLayout->addSubItem(btnDelete);

	// Run file/url  layout
	GLinearLayout* runTestLayout = new GLinearLayout("runTestLayout");
	runTestLayout->setOrientation(GLinearLayout::HORIZONTAL);
	leftSideLayout->addSubItem(runTestLayout);

	// sample type dropdown
	ddTestDataSourceType = new RUDropdown();
	ddTestDataSourceType->setWidth(120);
	ddTestDataSourceType->setHeight(30);
	ddTestDataSourceType->setOptionsShown(2);
	ddTestDataSourceType->setName("ddTestDataSourceType");
	runTestLayout->addSubItem(ddTestDataSourceType);

	ddTestDataSourceType->addOption(" File ");
	ddTestDataSourceType->addOption(" URL  ");

	//  sample path textbox
	tbTestDataSourcePath = new RUTextbox();
	tbTestDataSourcePath->setWidth(300);
	tbTestDataSourcePath->setHeight(30);
	tbTestDataSourcePath->setText("datasets/");
	tbTestDataSourcePath->setName("tbTestDataSourcePath");
	runTestLayout->addSubItem(tbTestDataSourcePath);

	// Run Button
	RUButton* sendButton = new RUButton("green");
	sendButton->setWidth(100);
	sendButton->setHeight(30);
	sendButton->setText("     Run");
	sendButton->setMouseDownListener(&GPanel::GuiCommander4);
	sendButton->setName("sendButton");
	runTestLayout->addSubItem(sendButton);

	// Kill Button
	RUButton* killButton = new RUButton("red");
	killButton->setWidth(100);
	killButton->setHeight(30);
	killButton->setText("     Kill");
	killButton->setMouseDownListener(&GPanel::GuiCommander13);
	killButton->setName("killButton");
	runTestLayout->addSubItem(killButton);

	// Cross Validation Layout
	GLinearLayout* crossValLayout = new GLinearLayout("crossValLayout");
	crossValLayout->setOrientation(GLinearLayout::HORIZONTAL);
	leftSideLayout->addSubItem(crossValLayout);

	// Cross Validation checkbox
	chkCrossVal = new RUCheckbox(" Cross Validate");
	chkCrossVal->setWidth(200);
	chkCrossVal->setHeight(30);
	chkCrossVal->setName("chkCrossVal");
	chkCrossVal->setCheck(true);
	chkCrossVal->setMouseDownListener(&GPanel::GuiCommander12);
	crossValLayout->addSubItem(chkCrossVal);

	// train % label
	lblttv = new RULabel();
	lblttv->setWidth(175);
	lblttv->setHeight(30);
	lblttv->setText("Train/Test/Val");
	lblttv->setName("lblttv");
	crossValLayout->addSubItem(lblttv);

	// train %
	tbTrainPct = new RUTextbox();
	tbTrainPct->setWidth(80);
	tbTrainPct->setHeight(30);
	tbTrainPct->setText("70");
	tbTrainPct->setName("tbTrainPct");
	crossValLayout->addSubItem(tbTrainPct);

	// test %
	tbTestPct = new RUTextbox();
	tbTestPct->setWidth(80);
	tbTestPct->setHeight(30);
	tbTestPct->setText("20");
	tbTestPct->setName("tbTestPct");
	crossValLayout->addSubItem(tbTestPct);

	// validation %
	tbValidationPct = new RUTextbox();
	tbValidationPct->setWidth(80);
	tbValidationPct->setHeight(30);
	tbValidationPct->setText("10");
	tbValidationPct->setName("tbValidationPct");
	crossValLayout->addSubItem(tbValidationPct);

	//============RIGHT============

	// Right Side forms
	GLinearLayout* rightSideLayout = new GLinearLayout("rightSideLayout");
	rightSideLayout->setX(Graphics::getWidth() - 500);
	rightSideLayout->setY(45);
	rightSideLayout->setOrientation(GLinearLayout::VERTICAL);
	rightSideLayout->setPadding(4);
	addSubItem(rightSideLayout);

	// Layer Size header
	lblLayerSize = new RULabel();
	lblLayerSize->setWidth(160);
	lblLayerSize->setHeight(40);
	lblLayerSize->setText("Layer Size");
	lblLayerSize->setName("lblLayerSize");
	rightSideLayout->addSubItem(lblLayerSize);

	GLinearLayout* hlcLayout = new GLinearLayout("hlcLayout");
	hlcLayout->setOrientation(GLinearLayout::HORIZONTAL);
	rightSideLayout->addSubItem(hlcLayout);

	// hidden layer count label
	lblHiddenLayerCount = new RULabel();
	lblHiddenLayerCount->setWidth(250);
	lblHiddenLayerCount->setHeight(30);
	lblHiddenLayerCount->setPadding(10);
	lblHiddenLayerCount->setText("Hidden Layer Count");
	lblHiddenLayerCount->setName("lblHiddenLayerCount");
	hlcLayout->addSubItem(lblHiddenLayerCount);

	// hidden layer count textbox
	tbHiddenLayerCount = new RUTextbox();
	tbHiddenLayerCount->setWidth(160);
	tbHiddenLayerCount->setHeight(30);
	tbHiddenLayerCount->setText(shmea::GType::intTOstring(formInfo->numHiddenLayers()));
	tbHiddenLayerCount->setName("tbHiddenLayerCount");
	tbHiddenLayerCount->setLoseFocusListener(&GPanel::GuiCommander9);
	hlcLayout->addSubItem(tbHiddenLayerCount);

	// Edit Input Layer Header
	lblEditInputLayer = new RULabel();
	lblEditInputLayer->setWidth(250);
	lblEditInputLayer->setHeight(40);
	lblEditInputLayer->setText("Edit Input Layer");
	lblEditInputLayer->setName("lblEditInputLayer");
	rightSideLayout->addSubItem(lblEditInputLayer);

	GLinearLayout* pInputEditLayout = new GLinearLayout("pInputEditLayout");
	pInputEditLayout->setOrientation(GLinearLayout::HORIZONTAL);
	rightSideLayout->addSubItem(pInputEditLayout);

	// pInput label
	lblPInput = new RULabel();
	lblPInput->setWidth(250);
	lblPInput->setHeight(30);
	lblPInput->setText("Input Layer p ");
	lblPInput->setName("lblPInput");
	pInputEditLayout->addSubItem(lblPInput);

	// pInput textbox
	tbPInput = new RUTextbox();
	tbPInput->setWidth(160);
	tbPInput->setHeight(30);
	tbPInput->setText("0.0");
	tbPInput->setName("tbPInput");
	pInputEditLayout->addSubItem(tbPInput);

	GLinearLayout* batchUpdateLayout = new GLinearLayout("batchUpdateLayout");
	batchUpdateLayout->setOrientation(GLinearLayout::HORIZONTAL);
	rightSideLayout->addSubItem(batchUpdateLayout);

	// Batch label
	RULabel* lblBatchSize = new RULabel();
	lblBatchSize->setWidth(250);
	lblBatchSize->setHeight(30);
	lblBatchSize->setText("Batch Size ");
	lblBatchSize->setName("lblBatchSize");
	batchUpdateLayout->addSubItem(lblBatchSize);

	// Batch dropdown
	tbBatchSize = new RUTextbox();
	tbBatchSize->setWidth(160);
	tbBatchSize->setHeight(30);
	tbBatchSize->setText("1");
	tbBatchSize->setName("tbBatchSize");
	batchUpdateLayout->addSubItem(tbBatchSize);

	GLinearLayout* hlSelectLayout = new GLinearLayout("hlSelectLayout");
	hlSelectLayout->setOrientation(GLinearLayout::HORIZONTAL);
	rightSideLayout->addSubItem(hlSelectLayout);

	// edit hidden layer header
	lblEditHiddenLayer = new RULabel();
	lblEditHiddenLayer->setWidth(250);
	lblEditHiddenLayer->setHeight(30);
	lblEditHiddenLayer->setText("Edit Hidden Layer");
	lblEditHiddenLayer->setName("lblEditHiddenLayer");
	hlSelectLayout->addSubItem(lblEditHiddenLayer);

	// index to edit dropdown
	ddIndexToEdit = new RUDropdown();
	ddIndexToEdit->setWidth(160);
	ddIndexToEdit->setHeight(30);
	ddIndexToEdit->setOptionsShown(3);
	ddIndexToEdit->setName("ddIndexToEdit");
	ddIndexToEdit->setMouseDownListener(&GPanel::GuiCommander3);
	hlSelectLayout->addSubItem(ddIndexToEdit);

	GLinearLayout* hlSizeLayout = new GLinearLayout("hlSizeLayout");
	hlSizeLayout->setOrientation(GLinearLayout::HORIZONTAL);
	rightSideLayout->addSubItem(hlSizeLayout);

	// hidden layer size label
	lblHiddenLayerSize = new RULabel();
	lblHiddenLayerSize->setWidth(250);
	lblHiddenLayerSize->setHeight(30);
	lblHiddenLayerSize->setText("Size");
	lblHiddenLayerSize->setName("lblHiddenLayerSize");
	hlSizeLayout->addSubItem(lblHiddenLayerSize);

	// hidden layer size textbox
	tbHiddenLayerSize = new RUTextbox();
	tbHiddenLayerSize->setWidth(160);
	tbHiddenLayerSize->setHeight(30);
	tbHiddenLayerSize->setName("tbHiddenLayerSize");
	hlSizeLayout->addSubItem(tbHiddenLayerSize);

	GLinearLayout* lcLayout = new GLinearLayout("lcLayout");
	lcLayout->setOrientation(GLinearLayout::HORIZONTAL);
	rightSideLayout->addSubItem(lcLayout);

	// learning rate label
	lblLearningRate = new RULabel();
	lblLearningRate->setWidth(250);
	lblLearningRate->setHeight(30);
	lblLearningRate->setText("Learning Rate");
	lblLearningRate->setName("lblLearningRate");
	lcLayout->addSubItem(lblLearningRate);

	// learning rate textbox
	tbLearningRate = new RUTextbox();
	tbLearningRate->setWidth(160);
	tbLearningRate->setHeight(30);
	tbLearningRate->setName("tbLearningRate");
	lcLayout->addSubItem(tbLearningRate);

	GLinearLayout* mfLayout = new GLinearLayout("mfLayout");
	mfLayout->setOrientation(GLinearLayout::HORIZONTAL);
	rightSideLayout->addSubItem(mfLayout);

	// momentum factor label
	lblMomentumFactor = new RULabel();
	lblMomentumFactor->setWidth(250);
	lblMomentumFactor->setHeight(30);
	lblMomentumFactor->setText("Momentum Factor");
	lblMomentumFactor->setName("lblMomentumFactor");
	mfLayout->addSubItem(lblMomentumFactor);

	// momentum factor textbox
	tbMomentumFactor = new RUTextbox();
	tbMomentumFactor->setWidth(160);
	tbMomentumFactor->setHeight(30);
	tbMomentumFactor->setName("tbMomentumFactor");
	mfLayout->addSubItem(tbMomentumFactor);

	GLinearLayout* wdLayout = new GLinearLayout("wdLayout");
	wdLayout->setOrientation(GLinearLayout::HORIZONTAL);
	rightSideLayout->addSubItem(wdLayout);

	// weight decay label
	lblWeightDecay = new RULabel();
	lblWeightDecay->setWidth(250);
	lblWeightDecay->setHeight(30);
	lblWeightDecay->setText("Weight Decay");
	lblWeightDecay->setName("lblWeightDecay");
	wdLayout->addSubItem(lblWeightDecay);

	// weight decay textbox
	tbWeightDecay = new RUTextbox();
	tbWeightDecay->setWidth(160);
	tbWeightDecay->setHeight(30);
	tbWeightDecay->setName("tbWeightDecay");
	wdLayout->addSubItem(tbWeightDecay);

	GLinearLayout* pHiddenLayout = new GLinearLayout("pHiddenLayout");
	pHiddenLayout->setOrientation(GLinearLayout::HORIZONTAL);
	rightSideLayout->addSubItem(pHiddenLayout);

	// pHidden label
	lblPHidden = new RULabel();
	lblPHidden->setWidth(250);
	lblPHidden->setHeight(30);
	lblPHidden->setText("Hidden Layer p: ");
	lblPHidden->setName("lblPHidden");
	pHiddenLayout->addSubItem(lblPHidden);

	// pHidden textbox
	tbPHidden = new RUTextbox();
	tbPHidden->setWidth(160);
	tbPHidden->setHeight(30);
	tbPHidden->setName("tbPHidden");
	pHiddenLayout->addSubItem(tbPHidden);

	GLinearLayout* actTypeLayout = new GLinearLayout("actTypeLayout");
	actTypeLayout->setOrientation(GLinearLayout::HORIZONTAL);
	rightSideLayout->addSubItem(actTypeLayout);

	// activation functions label
	lblActivationFunctions = new RULabel();
	lblActivationFunctions->setWidth(250);
	lblActivationFunctions->setHeight(30);
	lblActivationFunctions->setText("Activation Type");
	lblActivationFunctions->setName("lblActivationFunctions");
	actTypeLayout->addSubItem(lblActivationFunctions);

	// activation functions textbox
	ddActivationFunctions = new RUDropdown();
	ddActivationFunctions->setWidth(160);
	ddActivationFunctions->setHeight(30);
	ddActivationFunctions->setOptionsShown(3);
	ddActivationFunctions->setName("ddActivationFunctions");
	actTypeLayout->addSubItem(ddActivationFunctions);

	ddActivationFunctions->addOption("Tanh");
	ddActivationFunctions->addOption("PWise Tanh");
	ddActivationFunctions->addOption("Sigmoid");
	ddActivationFunctions->addOption("PWise Sigmoid");
	ddActivationFunctions->addOption("Linear");
	ddActivationFunctions->addOption("ReLu");
	ddActivationFunctions->addOption("Leaky ReLu");

	GLinearLayout* actParamLayout = new GLinearLayout("actParamLayout");
	actParamLayout->setOrientation(GLinearLayout::HORIZONTAL);
	rightSideLayout->addSubItem(actParamLayout);

	// Activation param label
	RULabel* lblActivationParam = new RULabel();
	lblActivationParam->setWidth(250);
	lblActivationParam->setHeight(30);
	lblActivationParam->setText("Activation Param");
	lblActivationParam->setName("lblActivationParam");
	actParamLayout->addSubItem(lblActivationParam);

	// Activation param textbox
	tbActivationParam = new RUTextbox();
	tbActivationParam->setWidth(160);
	tbActivationParam->setHeight(30);
	tbActivationParam->setName("tbActivationParam");
	actParamLayout->addSubItem(tbActivationParam);

	GLinearLayout* hCopyLayout = new GLinearLayout("hCopyLayout");
	hCopyLayout->setOrientation(GLinearLayout::HORIZONTAL);
	rightSideLayout->addSubItem(hCopyLayout);

	// Layer Dest label
	RULabel* lblCopyDest = new RULabel();
	lblCopyDest->setWidth(250);
	lblCopyDest->setHeight(30);
	lblCopyDest->setText("Dest Layer");
	lblCopyDest->setName("lblCopyDest");
	hCopyLayout->addSubItem(lblCopyDest);

	// copy/remove destination textbox
	tbCopyDestination = new RUTextbox();
	tbCopyDestination->setWidth(160);
	tbCopyDestination->setHeight(30);
	tbCopyDestination->setName("tbCopyDestination");
	hCopyLayout->addSubItem(tbCopyDestination);

	GLinearLayout* bCopyLayout = new GLinearLayout("bCopyLayout");
	bCopyLayout->setOrientation(GLinearLayout::HORIZONTAL);
	rightSideLayout->addSubItem(bCopyLayout);

	// Copy button
	RUButton* btnCopy = new RUButton();
	btnCopy->setWidth(122);
	btnCopy->setHeight(30);
	btnCopy->setText("     Copy");
	btnCopy->setMouseDownListener(&GPanel::GuiCommander6);
	btnCopy->setName("btnCopy");
	bCopyLayout->addSubItem(btnCopy);

	// Remove button
	RUButton* btnRemove = new RUButton("red");
	btnRemove->setWidth(122);
	btnRemove->setHeight(30);
	btnRemove->setText("   Remove");
	btnRemove->setMouseDownListener(&GPanel::GuiCommander7);
	btnRemove->setName("btnRemove");
	bCopyLayout->addSubItem(btnRemove);

	// Empty label for padding
	lblEditOutputLayer = new RULabel();
	lblEditOutputLayer->setWidth(200);
	lblEditOutputLayer->setHeight(5);
	lblEditOutputLayer->setText("");
	lblEditOutputLayer->setName("lblEmpty");
	rightSideLayout->addSubItem(lblEditOutputLayer);

	// Edit Output Layer Header
	lblEditOutputLayer = new RULabel();
	lblEditOutputLayer->setWidth(275);
	lblEditOutputLayer->setHeight(40);
	lblEditOutputLayer->setText("Edit Output Layer");
	lblEditOutputLayer->setName("lblEditOutputLayer");
	rightSideLayout->addSubItem(lblEditOutputLayer);

	GLinearLayout* outputTypeLayout = new GLinearLayout("outputTypeLayout");
	outputTypeLayout->setOrientation(GLinearLayout::HORIZONTAL);
	rightSideLayout->addSubItem(outputTypeLayout);

	// output type label
	lblOutputType = new RULabel();
	lblOutputType->setWidth(200);
	lblOutputType->setHeight(30);
	lblOutputType->setText("Output Type");
	lblOutputType->setName("lblOutputType");
	outputTypeLayout->addSubItem(lblOutputType);

	// output type dropdown
	ddOutputType = new RUDropdown();
	ddOutputType->setWidth(210);
	ddOutputType->setHeight(30);
	ddOutputType->setOptionsShown(2);
	ddOutputType->setName("ddOutputType");

	ddOutputType->addOption(" Regression");
	ddOutputType->addOption(" Classification");
	ddOutputType->addOption(" KL Divergence");
	outputTypeLayout->addSubItem(ddOutputType);

	GLinearLayout* outputSizeLayout = new GLinearLayout("outputSizeLayout");
	outputSizeLayout->setOrientation(GLinearLayout::HORIZONTAL);
	rightSideLayout->addSubItem(outputSizeLayout);

	// output layer size label
	lblOutputLayerSize = new RULabel();
	lblOutputLayerSize->setWidth(200);
	lblOutputLayerSize->setHeight(30);
	lblOutputLayerSize->setText("Size");
	lblOutputLayerSize->setName("lblOutputLayerSize");
	outputSizeLayout->addSubItem(lblOutputLayerSize);

	// output layer size textbox
	tbOutputLayerSize = new RUTextbox();
	tbOutputLayerSize->setWidth(210);
	tbOutputLayerSize->setHeight(30);
	tbOutputLayerSize->setText("1");
	tbOutputLayerSize->setName("tbOutputLayerSize");
	outputSizeLayout->addSubItem(tbOutputLayerSize);

	//============END OF RIGHT============

	loadDDNN();
	populateIndexToEdit();
	populateHLayerForm();
}

void NNCreatorPanel::onStart()
{
	// printf("RUComponentContainer Size: %ld\n", ruComponentContainer.size());
}

void NNCreatorPanel::loadDDNN()
{
	// clear the old items
	ddNeuralNet->clearOptions();

	// add the "new" item option
	ddNeuralNet->addOption(" New");

	// add items from nnetworks table to dropdown
	shmea::SaveList* nnList = new shmea::SaveList("neuralnetworks");
	nnList->load();
	std::vector<shmea::SaveItem*> saveTables = nnList->getItems();
	for (unsigned int i = 0; i < saveTables.size(); ++i)
	{
		shmea::SaveItem* cItem = saveTables[i];
		if (!cItem)
			continue;

		ddNeuralNet->addOption(cItem->getName());
	}
}

/*!
 * @brief hidden layer textbox populator
 * @details fills in the Hidden Layer variable textboxes and dropdowns based on the
 * currentHiddenLayerIndex and hiddenLayers variables
 */
void NNCreatorPanel::populateHLayerForm()
{
	if (currentHiddenLayerIndex < 0 || currentHiddenLayerIndex >= formInfo->numHiddenLayers())
		return;

	HiddenLayerInfo* currentLayer = formInfo->getLayers()[currentHiddenLayerIndex];
	ddIndexToEdit->setSelectedIndex(currentHiddenLayerIndex);
	tbHiddenLayerSize->setText(shmea::GType::intTOstring(currentLayer->size()));

	if (currentLayer->getLearningRate())
		tbLearningRate->setText(shmea::GType::floatTOstring(currentLayer->getLearningRate()));
	else
		tbLearningRate->setText(shmea::GType::floatTOstring(currentLayer->getLearningRate()));

	if (currentLayer->getMomentumFactor())
		tbMomentumFactor->setText(shmea::GType::floatTOstring(currentLayer->getMomentumFactor()));
	else
		tbMomentumFactor->setText(shmea::GType::floatTOstring(currentLayer->getMomentumFactor()));

	if (currentLayer->getWeightDecay())
		tbWeightDecay->setText(shmea::GType::floatTOstring(currentLayer->getWeightDecay()));
	else
		tbWeightDecay->setText(shmea::GType::floatTOstring(currentLayer->getWeightDecay()));

	if (currentLayer->getPHidden())
		tbPHidden->setText(shmea::GType::floatTOstring(currentLayer->getPHidden()));
	else
		tbPHidden->setText(shmea::GType::floatTOstring(currentLayer->getPHidden()));

	if (currentLayer->getActivationParam())
		tbActivationParam->setText(shmea::GType::floatTOstring(currentLayer->getActivationParam()));
	else
		tbActivationParam->setText(shmea::GType::floatTOstring(currentLayer->getActivationParam()));

	ddActivationFunctions->setSelectedIndex(currentLayer->getActivationType());
}

/*!
 * @brief store hidden layer textbox values
 * @details stores the Hidden Layer Variable textbox values into
 * hiddenLayers[currentHiddenLayerIndex]
 */
void NNCreatorPanel::syncFormVar()
{
	// Input Stuff

	shmea::GType pInput(tbPInput->getText());
	formInfo->setPInput(pInput.getFloat());

	shmea::GType batchSize(tbBatchSize->getText());
	formInfo->setBatchSize(batchSize.getLong());

	// Output Stuff

	shmea::GType outputSize(tbOutputLayerSize->getText());
	formInfo->setOutputSize(outputSize.getFloat());

	shmea::GType outputType((int)ddOutputType->getSelectedIndex()); // this is probably fine...
	formInfo->setOutputType(outputType.getLong());

	if (currentHiddenLayerIndex < 0 || currentHiddenLayerIndex >= formInfo->numHiddenLayers())
		return;

	HiddenLayerInfo* currentLayer = formInfo->getLayers()[currentHiddenLayerIndex];

	shmea::GType pHidden(tbPHidden->getText());
	currentLayer->setPHidden(pHidden.getFloat());

	shmea::GType hLayerSize(tbHiddenLayerSize->getText());
	currentLayer->setSize((int)hLayerSize.getLong());

	shmea::GType learningRate(tbLearningRate->getText());
	currentLayer->setLearningRate(learningRate.getFloat());

	shmea::GType momentumFactor(tbMomentumFactor->getText());
	currentLayer->setMomentumFactor(momentumFactor.getFloat());

	shmea::GType weightDecay(tbWeightDecay->getText());
	currentLayer->setWeightDecay(weightDecay.getFloat());

	int activationType = ddActivationFunctions->getSelectedIndex();
	switch (activationType)
	{
	case 0:
	{
		currentLayer->setActivationType(GMath::TANH);
		break;
	}
	case 1:
	{
		currentLayer->setActivationType(GMath::TANHP);
		break;
	}
	case 2:
	{
		currentLayer->setActivationType(GMath::SIGMOID);
		break;
	}
	case 3:
	{
		currentLayer->setActivationType(GMath::SIGMOIDP);
		break;
	}
	case 4:
	{
		currentLayer->setActivationType(GMath::LINEAR);
		break;
	}
	case 5:
	{
		currentLayer->setActivationType(GMath::RELU);
		break;
	}
	case 6:
	{
		currentLayer->setActivationType(GMath::LEAKY);
		break;
	}
	default:
	{
		currentLayer->setActivationType(GMath::TANH);
		break;
	}
	}

	shmea::GType activationParam(tbActivationParam->getText());
	currentLayer->setActivationParam(activationParam.getFloat());
}
/*!
 * @brief IndexToEdit populator
 * @details populates the IndexToEdit dropdown, usually whenever the Hidden Layer Count textbox
 * changes
 */
void NNCreatorPanel::populateIndexToEdit(int newSelectedIndex)
{
	ddIndexToEdit->clearOptions();

	// needed to offset the clearOptions action
	for (int i = 0; i < formInfo->numHiddenLayers(); ++i)
		ddIndexToEdit->addOption(shmea::GType::intTOstring(i));

	if (newSelectedIndex < formInfo->numHiddenLayers())
		ddIndexToEdit->setSelectedIndex(newSelectedIndex);
}

/*!
 * @brief load neural net
 * @details load neural net variables into the form
 * @param skeleton neural network architecture
 */
void NNCreatorPanel::loadNNet(glades::NNInfo* info)
{
	formInfo = info;
	std::string netName = formInfo->getName();
	tbNetName->setText(netName);

	float pInput = formInfo->getPInput();
	tbPInput->setText(shmea::GType::floatTOstring(pInput));

	int batchSize = formInfo->getBatchSize();
	tbBatchSize->setText(shmea::GType::intTOstring(batchSize));

	currentHiddenLayerIndex = 0;
	tbHiddenLayerCount->setText(shmea::GType::intTOstring(formInfo->numHiddenLayers()));

	int outputSize = formInfo->getOutputLayerSize();
	int outputType = formInfo->getOutputType();
	tbOutputLayerSize->setText(shmea::GType::intTOstring(outputSize));
	ddOutputType->setSelectedIndex(outputType);

	populateHLayerForm();
	populateIndexToEdit(currentHiddenLayerIndex);

	// Display a popup alert
	char buffer[netName.length()];
	sprintf(buffer, "Loaded \"%s\"", netName.c_str());
	Graphics::MsgBox("Neural Net", buffer, RUMsgBox::MESSAGEBOX);
}

/*!
 * @brief parse Cross Val percentage parser
 * @details parses a percentage into a percent value
 * @param spct the string containing the percentage
 * @return the percentage between 0 and 100
 */
int64_t NNCreatorPanel::parsePct(const std::string& spct)
{
	shmea::GType pct(spct);
	if (pct.getType() != shmea::GType::LONG_TYPE)
		return -1l;

	int64_t pctVal = pct.getLong();
	if (pctVal < 0 || pctVal > 100)
		return -1l;

	return pctVal;
}

/*!
 * @brief Save NN
 * @details save a neural network using the Save_NN network service
 * @param cmpName the name of the component where the event happened
 * @param x the x-coordinate where the event happened (not used)
 * @param y the y-coordinate where the event happened (not used)
 */
void NNCreatorPanel::GuiCommander1(const std::string& cmpName, int x, int y)
{
	std::string serverIP = "127.0.0.1"; // 69.126.139.205

	// make sure the layers are up to date
	syncFormVar();

	std::string netName = tbNetName->getText();
	formInfo->setName(netName);

	shmea::GType pInput(tbPInput->getText());
	if (pInput.getType() != shmea::GType::FLOAT_TYPE)
		return;

	shmea::GType batchSize(tbBatchSize->getText());
	if (batchSize.getType() != shmea::GType::LONG_TYPE)
		return;

	shmea::GType outputSize(tbOutputLayerSize->getText());
	if (outputSize.getType() != shmea::GType::LONG_TYPE)
		return;

	// just make sure the expected hidden layer counts are equal
	shmea::GType layerCountCheck(tbHiddenLayerCount->getText());
	if (layerCountCheck.getType() != shmea::GType::LONG_TYPE)
		return;

	int newCount = (int)layerCountCheck.getLong();
	if (newCount != formInfo->numHiddenLayers())
		return;

	// Save the neural network
	populateHLayerForm();
	NNetwork* network = new NNetwork(formInfo);
	glades::saveNeuralNetwork(network);
	loadDDNN();
}

/*!
 * @brief Load NN
 * @param cmpName the name of the component where the event happened
 * @param x the x-coordinate where the event happened (not used)
 * @param y the y-coordinate where the event happened (not used)
 */
void NNCreatorPanel::GuiCommander2(const std::string& cmpName, int x, int y)
{
	//
}

/*!
 * @brief on Index to Edit change
 * @details runs when ddIndexToEdit changes value
 * @param cmpName the name of the component where the event happened
 * @param x the x-coordinate where the event happened (not used)
 * @param y the y-coordinate where the event happened (not used)
 */
void NNCreatorPanel::GuiCommander3(const std::string& cmpName, int x, int y)
{
	int indexToEdit = ddIndexToEdit->getSelectedIndex();
	if (indexToEdit == currentHiddenLayerIndex)
		return;

	syncFormVar();
	currentHiddenLayerIndex = indexToEdit;
	populateHLayerForm();
}

/*!
 * @brief event handler 4
 * @details handles a NNCreator event
 * @param cmpName the name of the component where the event happened
 * @param x the x-coordinate where the event happened
 * @param y the y-coordinate where the event happened
 */
void NNCreatorPanel::GuiCommander4(const std::string& cmpName, int x, int y)
{
	if (ddTestDataSourceType->getSelectedIndex() == (unsigned int)-1)
		return;

	int importType = shmea::GTable::TYPE_FILE;
	if (ddTestDataSourceType->getSelectedIndex() == 0)
	{
		// If File is selected
		importType = shmea::GTable::TYPE_FILE;
	}
	else if (ddTestDataSourceType->getSelectedIndex() == 1)
	{
		// If URL is selected
		importType = shmea::GTable::TYPE_URL;
	}

	std::string serverIP = "127.0.0.1"; // 69.126.139.205
	if (GNet::getServerInstanceList().find(serverIP) != GNet::getServerInstanceList().end())
	{
		GNet::Instance* cInstance = (GNet::getServerInstanceList().find(serverIP))->second;

		// Get the vars from the components
		std::string netName = tbNetName->getText();
		std::string testFName = tbTestDataSourcePath->getText();
		int64_t trainPct = parsePct(tbTrainPct->getText()),
				testPct = parsePct(tbTestPct->getText()),
				validationPct = parsePct(tbValidationPct->getText());

		if ((netName.length() == 0) || (testFName.length() == 0))
			return;

		if (trainPct == -1 || testPct == -1 || validationPct == -1 ||
			trainPct + testPct + validationPct != 100)
		{
			printf("Percentages invalid: \n\tTrain: %s \n\tTest: %s \n\tValidation: %s \n",
				   tbTrainPct->getText().c_str(), tbTestPct->getText().c_str(),
				   tbValidationPct->getText().c_str());
			return;
		}

		// Run a machine learning instance
		shmea::GList wData;
		wData.addString("ML_Train");
		wData.addString(netName);
		wData.addString(testFName);
		wData.addInt(importType);
		/*wData.addLong(trainPct);
		wData.addLong(testPct);
		wData.addLong(validationPct);*/
		GNet::Service::ExecuteService(wData, cInstance);
	}
}

/*!
 * @brief
 * @details switches the active UI panel
 * @param cmpName the name of the component where the event happened
 * @param x the x-coordinate where the event happened
 * @param y the y-coordinate where the event happened
 */
void NNCreatorPanel::GuiCommander5(const std::string& cmpName, int x, int y)
{
	//
}

/*!
 * @brief event handler 6
 * @details handles a NNCreator event
 * @param cmpName the name of the component where the event happened
 * @param x the x-coordinate where the event happened
 * @param y the y-coordinate where the event happened
 */
void NNCreatorPanel::GuiCommander6(const std::string& cmpName, int x, int y)
{
	syncFormVar();
	shmea::GType destination(tbCopyDestination->getText());
	if (destination.getType() != shmea::GType::LONG_TYPE)
		return;

	int dst = destination.getInt();
	if (dst < 0 || dst >= formInfo->numHiddenLayers() || dst == currentHiddenLayerIndex)
		return;

	formInfo->copyHiddenLayer(dst, currentHiddenLayerIndex);
	printf("[GUI] Layer %d parameters copied to layer %d\n", currentHiddenLayerIndex, dst);
}

/*!
 * @brief event handler 7
 * @details handles a NNCreator event
 * @param cmpName the name of the component where the event happened
 * @param x the x-coordinate where the event happened
 * @param y the y-coordinate where the event happened
 */
void NNCreatorPanel::GuiCommander7(const std::string& cmpName, int x, int y)
{
	if (formInfo->numHiddenLayers() <= 1)
		return;

	formInfo->removeHiddenLayer(currentHiddenLayerIndex);
	if (currentHiddenLayerIndex > formInfo->numHiddenLayers())
		--currentHiddenLayerIndex;

	printf("[GUI] Layer %d deleted\n", currentHiddenLayerIndex);
	populateHLayerForm();
	tbHiddenLayerCount->setText(shmea::GType::intTOstring(formInfo->numHiddenLayers()));
	populateIndexToEdit(currentHiddenLayerIndex);
}

/*!
 * @brief event handler 8
 * @details handles a NNCreator event
 * @param cmpName the name of the component where the event happened
 * @param x the x-coordinate where the event happened
 * @param y the y-coordinate where the event happened
 */
void NNCreatorPanel::GuiCommander8(const std::string& cmpName, int x, int y)
{
	//
}

/*!
 * @brief Hidden Layer Count change handler
 * @details runs whenever the tbHiddenLayerCount changes (i.e. it loses focus). For now, changes
 * hiddenLayers, currentHiddenLayerIndex, and formInfo->numHiddenLayers() accordingly
 */
void NNCreatorPanel::GuiCommander9()
{
	shmea::GType newHiddenLayerCount(tbHiddenLayerCount->getText());
	if (newHiddenLayerCount.getType() != shmea::GType::LONG_TYPE)
		return;

	int newCount = (int)newHiddenLayerCount.getLong();
	if (newCount < 0 || (newCount == formInfo->numHiddenLayers()))
		return;

	formInfo->resizeHiddenLayers(newCount);
	if (currentHiddenLayerIndex >= newCount)
	{
		currentHiddenLayerIndex = formInfo->numHiddenLayers() - 1;
		populateHLayerForm();
	}

	populateIndexToEdit(currentHiddenLayerIndex);
}

void NNCreatorPanel::GuiCommander10(const std::string& cmpName, int x, int y)
{
	loadDDNN();
}

void NNCreatorPanel::GuiCommander11(char eventKeyPressed)
{
	//
}

void NNCreatorPanel::GuiCommander12(const std::string& cmpName, int x, int y)
{
	// Hide cross validate options if visible
	if (lblttv->isVisible())
	{
		lblttv->setVisible(false);
		tbTrainPct->setVisible(false);
		tbTestPct->setVisible(false);
		tbValidationPct->setVisible(false);
	}
	else // Show cross validate options if hidden
	{
		lblttv->setVisible(true);
		tbTrainPct->setVisible(true);
		tbTestPct->setVisible(true);
		tbValidationPct->setVisible(true);
	}
}

void NNCreatorPanel::GuiCommander13(const std::string& cmpName, int x, int y)
{
	NNetwork::caboose = true;
}

void NNCreatorPanel::GuiCommander14(const std::string& cmpName, int x, int y)
{
	std::string netName = tbNetName->getText();
	shmea::SaveList* nnList = new shmea::SaveList("neuralnetworks");
	nnList->deleteItem(netName);
	loadDDNN();

	// Display a popup alert
	char buffer[netName.length()];
	sprintf(buffer, "Deleted \"%s\"", netName.c_str());
	Graphics::MsgBox("Neural Net", buffer, RUMsgBox::MESSAGEBOX);
}

void NNCreatorPanel::GuiCommander15(int newIndex)
{
	// Only load after a selection click (not an open click)
	if (ddNeuralNet->isOpen())
		return;

	// make sure the user wants to actually load a network, since index 0 = "New"
	int loadOrSave = ddNeuralNet->getSelectedIndex();
	if (loadOrSave == 0)
		return;

	std::string netName = ddNeuralNet->getSelectedText();
	tbNetName->setText(netName);
	formInfo->setName(netName);

	// Load a machine learning model
	NNetwork* cNetwork = glades::getNeuralNetwork(netName);

	// Load the NN into the form
	loadNNet(cNetwork->getNNInfo());
}

/*!
 * @brief Plot the Learning Curve of the NN
 * @details Plot the learning curve of the neural network on the graph
 */
void NNCreatorPanel::PlotLearningCurve(const shmea::GList& graphPoints)
{
	lcGraph->setLine("lc", graphPoints);
}

/*!
 * @brief Plot the ROC Curve of the NN
 * @details Plot the ROC Curve of the neural network on the graph
 */
void NNCreatorPanel::PlotROCCurve(const std::vector<Point2*>& graphPoints)
{
	rocCurveGraph->setPoints("lc", graphPoints);
}

/*!
 * @brief Plot the Expected vs Predicted of the NN
 * @details Plot the Expected vs Predicted of the neural network on the proper graph
 */
void NNCreatorPanel::PlotScatter(const shmea::GTable& graphPoints)
{
	dartboardGraph->addScatterPoints(graphPoints);
}

void NNCreatorPanel::updateConfMatrixTable(const shmea::GTable& confMatrix)
{
	cMatrixTable->import(confMatrix);
}
