/*
 * CNeuralNet.h
 *
 *  Created on: 26 Dec 2013
 *      Author: benjamin
 */

#ifndef CNEURALNET_H_
#define CNEURALNET_H_
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdlib.h>
#include <cstring>
#include <stdio.h>
#include <stdint.h>

typedef unsigned int uint;

/****************************************
** --> Artificial neuron structure <-- **
*****************************************/
struct Neuron
{
	// number of inputs into the neuron
	int numInputs;

	// weights for each input
	std::vector<double> weightVector;

	// store the error of the node --> to simplify back propagation
	double n_error;

	// store the output of a neuron to calculate errors
	double n_output;

	//store the previous delta weight used to calculate the momentum term
	std::vector<double> previousDeltas ;

	// Neuron Constructor
	Neuron(int numberInputs);
};

/************************************************
** --> Struct to define a layer of neurons <-- **
*************************************************/
struct NeuronLayer
{
	// number of neurons in this layer
	int numNeurons;

	// layer (vector) of neurons
	std::vector<Neuron> neuronVector;

	// Neuron layer constructor
	NeuronLayer(int numberNeurons, int numInputsPerNeuron);
};

class CNeuralNet 
{
private:
	uint _inputLayerSize;
	uint _hiddenLayerSize;
	uint _outputLayerSize;
	double _lRate;
	double _mse_cutoff;
	double momentum = 0.9;
	double MSE = 1;			// Mean Squared Error value
	double MSEv = 1;        // Mean Squared Error value for the validation set
		 
	std::vector<NeuronLayer> layersVector; // Storage for each layer of neurons including the output layer	
	std::vector<double> outputsVector; // Storage for the output layers calculated output

protected:
	void feedForward(std::vector<double> inputs); //you may modify this to do std::vector<double> if you want
	void propagateErrorBackward(std::vector<double> desiredOutput, std::vector<double> inputsLayer); //you may modify this to do std::vector<double> if you want
	double meanSquaredError(std::vector<double> desiredOutput); //you may modify this to do std::vector<double> if you want
public:
	CNeuralNet(uint inputLayerSize, uint hiddenLayerSize, uint outputLayerSize, double lRate, double mse_cutoff);
	void initWeights();
	void train(std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> outputs, uint trainingSetSize); //you may modify this to do std::vector<std::vector<double> > or do boost multiarray or something else if you want
	uint classify(std::vector<double> input); //you may modify this to do std::vector<double> if you want
	double getOutput(uint index) const;
	virtual ~CNeuralNet();
};

#endif /* CNEURALNET_H_ */
