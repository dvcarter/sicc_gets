#include <iostream>
#include <time.h>
#include <cmath>
#include <iomanip>
#include "setup/doublesetup_functions.cpp"
#include "setup/doublelstm_functions.cpp"
#include "setup/parseCSV.cpp"
// rows
#define H 5
// cols
#define J 500
#define OK 4
// U cols
#define K 1

using namespace std;

// name your terms
// a_t input activation
// x_t sequential data
//


// simplification
// C = AB
// A = ixj
// B = jxp
// Weight[H][J] * input[J][1]

// vanilla forward pass without bias

// vanilla backward propagation


// LSTM forward components

// internal state: ((*) = outer product)
//	state_t = a_t (*) i_t + f_t (*) state_t-1

// output:
//	out_t = tanh(state_t) (*) o_t


// LSTM backward propagation components
//	d_out_t = Delta_t + Delta_out_t

//	d_state_t = d_out_t (*) o_t (*) (1 - tanh^2(state_t)) + d_state_t+1 (*) f_t+1
//


int main() {

/*******************************************************************************************************************/
	//  training, testing, and learning rate decisions
	int trainingEpoch;
	int trainingAmount;
	int windowSize;
	int testAmount;
	double rho;
	int sequenceAmount;

	cout << "Enter epoch amount (positive integer): " <<endl;
	cin >> trainingEpoch;
	cout << "Enter training amount (positive integer) [batch size]: " <<endl;
	cin >> trainingAmount;
	cout << "Enter window size (number of days) :" << endl;
	cin >> windowSize;
	cout << "Enter test Amount amount (positive integer): " <<endl;
	cin >> testAmount;
	cout << "Enter learning rate used for weight updating (real number): ";
	cin >> rho;

	
/*******************************************************************************************************************/
	//setup data
	// mock sequential data
	// x_t[1][J];
	// x_0
	
	// trainingData
	vector<vector<double> > inputData;
	
	// testingData
	vector<vector<double> > amazonData;

	double targetData[2516][1];
	parseCSV(&inputData, "data/inputData.csv");
	parseCSV(&amazonData, "data/AMZN.csv");

	double x_t[inputData.size()];
	initializeArray(inputData.size(), x_t, 0.0);

	double target_t[inputData.size()];
	
	// validation output
        double * output_t = new double[trainingAmount];
	initializeArray(trainingAmount, output_t, 0.0);

	// testData holder
	double x_Test[amazonData.size()];
	initializeArray(amazonData.size(), x_Test, 0.0);

	double target_Test[amazonData.size()];
	initializeArray(amazonData.size(), target_Test, 0.0);

	double * output_Test = new double[windowSize];
	initializeArray(windowSize, output_Test, 0.0);

        // initialize weight matrices
        matrix <double>  w(4, OK, 0.0);
        matrix <double> u( 4,K, 0.0);

        // populate initial input weights
        srand(time(NULL));
        weightInitializerInput(4, OK, &w);
        printMatrix(w, "Input Weights");
	
        double w_a[OK];
        double w_i[OK];
        double w_f[OK];
	double w_o[OK];
	initializeArray(OK, w_a, 0.0);
	initializeArray(OK, w_i, 0.0);
	initializeArray(OK, w_f, 0.0);
	initializeArray(OK, w_o, 0.0);

        matrixToArray(w, w_a, w_i, w_f, w_o);

        // populate initial output weights
//        srand(time(NULL));
        weightInitializerOutput(4, 1, &u);
        printMatrix(u, "Output Weights");

	double u_a[K];
	double u_i[K];
	double u_f[K];
	double u_o[K];
	initializeArray(K, u_a, 0.0);
	initializeArray(K, u_i, 0.0);
	initializeArray(K, u_f, 0.0);
	initializeArray(K, u_o, 0.0);

	matrixToArray(u, u_a, u_i, u_f, u_o);

        matrix<double> uT(K,4, 0.0);
	//transposeMatrix(u, &uT);
	//printMatrix(uT, "output (prediction) matrix transposed");
	// bias is gathered later in order to deal with the fact that the forget gate requires a different initial value

/*******************************************************************************************************************/
	// initialize input activation
	//float a_t[OK];
	double * a_t = new double[trainingAmount];
	double W_a[H] = {0.0};
	double U_a[H] = {0.0};
	double b_a[H] = {0.0};
	double a_weight_sum[H];

	// initial gate setup
	 gateSetup(trainingAmount, H, a_t, W_a, U_a, b_a, a_weight_sum, output_t, "input activation");

	// forward input gate
	//float i_t[OK];
	double * i_t = new double[trainingAmount];
	double W_i[H];
	double U_i[H];
	double b_i[H];
	double i_weight_sum[H];
	gateSetup(trainingAmount, H, i_t, W_i, U_i, b_i, i_weight_sum, output_t, "input gate");
	
	// forget gate
	//float f_t[OK];
	double * f_t = new double[trainingAmount];
	double W_f[H];
	double U_f[H];
	double b_f[H];
	double f_weight_sum[H];

        // forget gate  initializer 
	gateSetup(trainingAmount, H, f_t, W_f, U_f, b_f, f_weight_sum, output_t, "forget gate");
	initializeArray(H, b_f, 1.0);    // initialize biases to 1 as per Jozefowicz, Zaremba, and Sutskever

	// output gate
        //float o_t[OK];
	double * o_t = new double[trainingAmount];
	double W_o[H];
	double U_o[H];
	double b_o[H];
	double o_weight_sum[H];

	// output gate initializer 
	gateSetup(trainingAmount, H,o_t, W_o, U_o, b_o, o_weight_sum, output_t, "output gate");
	
	// state
	// internal state: ((*) = outer product)
	// state_t = a_t (*) i_t + f_t (*) state_t-1

	//float state_t[OK] = {0.0};
	double * state_t = new double[trainingAmount];
	initializeArray(trainingAmount, state_t, 0.0);

	// gates
	matrix <double>  gates_t(4, trainingAmount, 0.0);
	matrix <double>  b(4,1, 0.0);


	buildMatrix(1, &b, b_a, b_i, b_f, b_o); 
//        printMatrix(b, "Bias in Matrix form");	
	
	matrixToArray(b, b_a, b_i, b_f, b_o);
//      printArray(1, b_a, "b a");
//      printArray(1, b_i, "b i ");
//      printArray(1, b_f, "b f");
//      printArray(1, b_o, "b o");



	// testing
	double * prediction_Test = new double[testAmount];
	double * a_Test = new double[windowSize];
	double * i_Test = new double[windowSize];
	double * f_Test = new double[windowSize];
	double * o_Test = new double[windowSize];
	double * state_Test = new double[windowSize];
	initializeArray(testAmount, prediction_Test, 0.0);
	initializeArray(windowSize, a_Test, 0.0);
	initializeArray(windowSize, i_Test, 0.0);
	initializeArray(windowSize, f_Test, 0.0);
	initializeArray(windowSize, o_Test, 0.0);
	initializeArray(windowSize, state_Test, 0.0);
/*******************************************************************************************************************/
	// backpropagation through time
	// i.e. updating weights 8 equations
	
	// state delta
	//float deltaState[OK];
	double * deltaState = new double[trainingAmount];
	
	//  output_t-1 delta
	//float deltaOutput[OK];
	double * deltaOutput = new double[trainingAmount];
	
	//float dOutput[OK];
	double * dOutput = new double[trainingAmount];

	// gate deltas 
	//float deltaA_t[OK];
	double * deltaA_t = new double[trainingAmount];

	//float deltaI_t[OK];
	double * deltaI_t = new double[trainingAmount];

	//float deltaF_t[OK];
	double * deltaF_t = new double[trainingAmount];

	//float deltaO_t[OK];
	double * deltaO_t = new double[trainingAmount];

	// hidden weight deltas
	matrix<double> deltaW_t(4, 4, 0.0);
//	printMatrix2(4, 4, deltaW_t, "Delta Input (Hidden) Weight (a, i, f, o) in Matrix form");
	double  deltaW_a[OK];
	double  deltaW_i[OK];
	double  deltaW_f[OK];
	double  deltaW_o[OK];

	// output weight deltas
	matrix<double> deltaU_t(4,1, 0.0);
//	printMatrix(deltaU_t, "Delta Output Weight (a, i, f, o) in Matrix form");
	double  deltaU_a[K];
	double  deltaU_i[K];
	double  deltaU_f[K];
	double  deltaU_o[K];

	// bias deltas
	matrix<double> deltaB_t(4,1, 0.0);
//	printMatrix(deltaB_t, "Delta Bias (a, i, f, o) in Matrix form");
	double  deltaB_a[K];
	double  deltaB_i[K];
	double  deltaB_f[K];
	double  deltaB_o[K];

	// initialize deltas to zero
	initializeArray(trainingAmount, deltaState, 0.0);
	initializeArray(trainingAmount, deltaOutput, 0.0);
	initializeArray(trainingAmount, dOutput, 0.0);
	initializeArray(trainingAmount, deltaA_t, 0.0);
	initializeArray(trainingAmount, deltaI_t, 0.0);
	initializeArray(trainingAmount, deltaF_t, 0.0);
	initializeArray(trainingAmount, deltaO_t, 0.0);
	initializeArray(OK, deltaW_a, 0.0);
	initializeArray(OK, deltaW_i, 0.0);
	initializeArray(OK, deltaW_f, 0.0);
	initializeArray(OK, deltaW_o, 0.0);
	initializeArray(K, deltaU_a, 0.0);
	initializeArray(K, deltaU_i, 0.0);
	initializeArray(K, deltaU_f, 0.0);
	initializeArray(K, deltaU_o, 0.0);
	initializeArray(K, deltaB_a, 0.0);
	initializeArray(K, deltaB_i, 0.0);
	initializeArray(K, deltaB_f, 0.0);
	initializeArray(K, deltaB_o, 0.0);
	double uTransposed[4];
	initializeArray(4, uTransposed, 0.0);
/*******************************************************************************************************************/
        matrix <double>  deltaGates(4, trainingAmount, 0.0);
//	printMatrix(deltaGates, "DeltaGates clean");
/*******************************************************************************************************************/
	// epochs forward passes
	double * loss_t = new double[trainingAmount];
	initializeArray(windowSize, loss_t, 0.0);
	for (int epoch = 0; epoch < trainingEpoch; epoch++) {
		cout << "Epoch level " << epoch << endl;
		
		// training loop
		// reset loss_holder after every epoch
		double loss_holder = 0.0;
		for (int t = 0; t < trainingAmount; t++) {
			cout << "training level level " << t << endl;
			
			// input values
			for (int i =0, j = t ; i <= windowSize; i++, j++) {
				x_t[i] = inputData[j][0];
			}
			// target values	
			target_t[t] = inputData[t+windowSize-1][2];
			// gates
				double sumHold;
				if(t == 0) {
					sumHold =0.0;
					for (int j = 0; j < OK; j++) {
						for (int k=0; k < windowSize; k++) {
							sumHold += w_a[j]*x_t[k];
						}
					}
					a_t[t] = tanh(sumHold);
				}	
				double sumHoldA_t;
				if (t > 0) {
					sumHoldA_t = 0.0;
					for (int j = 0; j < OK; j++) {
						for(int k = 0; k < windowSize; k++) {
							sumHoldA_t += w_a[j]*x_t[k]  + u_a[0]*output_t[t-1] + b_a[0];
						}

					}
					a_t[t] = tanh(sumHoldA_t);
				}
			// input gate
			// i_t = g(W_i * x_t + U_i * out_t-1 + b_i)
			  cout << "input values used for input gate at training level: " << t <<endl;
				double sumI_t;
				if (t == 0) {
					sumI_t = 0.0;
					for (int j = 0; j < OK; j++) {
						for (int k = 0; k < windowSize; k++) {
							sumI_t += w_i[j]*x_t[k];
						}
					}
					i_t[t] = sigmoid_proper(sumI_t+b_i[0]);	
				}
				double sumI_t2;
				if ( t > 0) {
					sumI_t2 = 0.0;
					for (int j = 0; j < OK; j++) {
						for (int k = 0; k < windowSize; k++) {
							sumI_t2 += w_i[j]*x_t[k];
							//sumI_t2 += w_i[j]*x_t[k]+u_i[0]*output_t[t-1]+b_i[0];
						}	
					}	
					i_t[t] = sigmoid_proper(sumI_t2+u_i[0]*output_t[t-1]+b_i[0]);	
				}	

			// forget gate
				double sumF_t;
				if (t == 0) {
					sumF_t = 0.0;
					for (int j = 0; j < OK; j++) {
						for (int k = 0; k < windowSize; k++) {
							sumF_t += w_f[j]*x_t[k];
							//sumF_t += w_f[j]*x_t[k]+b_f[0];
						}
					}
					cout << endl;
					f_t[t] = sigmoid_proper(sumF_t+b_f[0]);
				}
				double sumF_t2;
				if (t > 0) {
					sumF_t2= 0.0;
					for (int j = 0; j < OK; j++) {
						for (int k = 0; k < windowSize; k++) {
							sumF_t2 += w_f[j]*x_t[k];
							//sumF_t2 += w_f[j]*x_t[k]+u_f[0]*output_t[t-1]+b_f[0];
						}
					}
					f_t[t] = sigmoid_proper(sumF_t2+u_f[0] *output_t[t-1] + b_f[0]);
				}
		
		
		
			// output gate
				double sumO_t;
				if (t == 0) {
					sumO_t = 0.0;
					for (int j = 0; j < OK; j++) {
						for (int k = 0; k< windowSize; k++) {
							sumO_t += w_o[j]*x_t[k]+b_o[0];
						} 
					}
					o_t[t] = sigmoid_proper(sumO_t);
				}
				double sumO_t2;
				if( t > 0) {
					sumO_t2 = 0.0;
					for (int j = 0; j < OK; j++) {
						for (int k=0;k < windowSize; k++) {
							sumO_t2 += w_o[j]*x_t[k]+u_o[0]*output_t[t-1]+b_o[0];
						}
					}
					o_t[t] = sigmoid_proper(sumO_t2);
				}		

	                // state
				if (t == 0) {
					state_t[t] = (i_t[t] * a_t[t]);
				}  
				if( t >= 1) {
					state_t[t] = (state_t[t-1] * f_t[t]) + (i_t[t] * a_t[t]);
				}

			// output
			output_t[t] = o_t[t] * state_t[t];
			// gates
			buildMatrix(trainingAmount, &gates_t, a_t, i_t, f_t, o_t);
			// deltas
			        
			// delta output
			deltaOutput[t] = loss_holder + dOutput[t];
			// delta state
			//	for (int i = 0, j = i+1; i < OK; i++) {
				if (epoch == 0) {
					deltaState[t] = deltaOutput[t] * o_t[t]*(1-(tanh(state_t[t])*tanh(state_t[t])));
				}
				if (epoch >=1) { 
					deltaState[t] = deltaOutput[t] * o_t[t]*(1-(tanh(state_t[t])*tanh(state_t[t]))) + deltaState[t+1]*f_t[t+1];
				}
				//	}
			// delta input activation
			deltaA_t[t] = deltaState[t]*i_t[t]*(1-(a_t[t]*a_t[t]));
			// delta input gate
			deltaI_t[t] = deltaState[t]*a_t[t]*i_t[t]*(1-i_t[t]);
			//delta forget gate
				if (t == 0) {
					deltaF_t[t] = 0;
				} 
				if (t > 0)
				{
					 deltaF_t[t] = deltaState[t]*state_t[t-1]*f_t[t]*(1-f_t[t]);
				}
			// delta output gate
			deltaO_t[t] = deltaOutput[t]*tanh(state_t[t])*o_t[t]*(1-o_t[t]);
			buildMatrix(trainingAmount, &deltaGates, deltaA_t, deltaI_t, deltaF_t, deltaO_t);
			
			matrixToArraySimplified2(OK, u, uTransposed);
			if (t > 0) {
				for (int i = 0; i < 4; i++) {
					dOutput[t-1] += uTransposed[i] * deltaGates[i][t];
				}
			}
			printArray(t, a_t, "input activation");
			printArray(t, i_t, "input gate");
			printArray(t, f_t, "forget gate");
			printArray(t, o_t, "output gate");
			printArray(t, state_t, "state gate");
/*******************************************************************************************************************/
			// delta W_t
			for (int i = 0; i < 4; i++) {
				for (int j = 0; j < 4; j++) {
					for (int k = 0; k < windowSize;k++) {
						deltaW_t[i][j] += deltaGates[i][t] * x_t[k];
					}
				}
			}
			if (t > 0) {
				for (int i = 0; i < 4; i++) {
					cout << "delta gates: "	<< deltaGates[i][t-1] << endl;
				}
			}
//
			// delta U_t
			for (int i = 0; i < 4; i++) {// windowSize+t
				deltaU_t[i][0] += deltaGates[i][t] * output_t[t];
			}

			// delta B_t
			for (int i = 0; i < 4; i++) {
			       for (int j = 0, k=t; j < 1; j++, k++) {
					deltaB_t[i][j] += deltaGates[i][k];
				}	
			}	       

/****************************************************************************************************************/
			// updating weights
			// delta W_t * rho
			//matrixByScalar(4,  &deltaW_t, rho);
			
			// w = w - rho*deltaW_t
 			printMatrix(deltaW_t, "delta Weights before");
			for (int i = 0; i < 4; i++) {
 				for (int j = 0, k=t; j < 4; j++, k++) {
 					w[i][j] = w[i][j] - (rho*deltaW_t[i][j]);
 				}
 			}
			// convert matrix back to arrays for testing
			matrixToArray(w, w_a, w_i, w_f, w_o);
			printMatrix(deltaW_t, "delta Weights after");
			printMatrix(w, "Updated Weights");

			// delta U_t * rho
			//matrixByScalar(1,  &deltaU_t, rho);

			// u = u - rho*deltaU_t
			printMatrix(deltaU_t, "delta U Weights before");
			for (int i = 0; i < 4; i++) {
				for (int j = 0, k=t; j < 1; j++) {
					u[i][j] = u[i][j] - (rho*deltaU_t[i][j]);
				}
			}
			// convert matrix back to arrays for testing
			printMatrix(deltaU_t, "delta U Weights after");
			matrixToArray(u, u_a, u_i, u_f, u_o);
			printMatrix(u, "Updated Output Weights");

			// delta B_t * rho
			//matrixByScalar(1,  &deltaB_t, rho);

			// b = b - rho*deltaB_t
			for (int i = 0; i< 4; i++) {
					b[i][0] = b[i][0] - (rho*deltaB_t[i][0]);
			}
			// convert matrix back to arrays for testing
			matrixToArray(b, b_a, b_i, b_f, b_o);
			printMatrix(b, "Updated Bias");
/****************************************************************************************************************
*/

			// // loss, where output_t is compared with target value
			// L2 Cost 
			// there is also an L1 cost
			// deltaL version
//				loss_holder = 0.5*((output_t[t] - target_t[t])*(output_t[t] - target_t[t]));
				loss_holder = (target_t[t]-output_t[t]);
				loss_t[t] = loss_holder;
		//
		
		
		}
	}
		// counter
		int count = 0;
		// testing section
		FILE * predictionSheet;
		predictionSheet = fopen("amazonPoints.txt", "w");
		for (int p = 0; p < testAmount; p++) {
			// input values
                        for (int i =0, j = p ; i < testAmount ; i++, j++) {
                                x_Test[i] = amazonData[j][0];
                        }   
                        printArray(testAmount, x_Test, "Input Values");
    
                        // target values        
			for (int i = 0, j=p; i <= testAmount; i++, j++) {				
				target_Test[i] = amazonData[j][3];
			}
			printArray(testAmount, target_Test, "Target Values");
			// prediction values
			for (int i = 0, j =p; i <= testAmount; i++, j++) {
				prediction_Test[i] = amazonData[j][2];
			}
			// gates
			for (int t = 0; t < windowSize; t++) {
				double sumHold;
				if(t == 0) {
					sumHold =0.0;
					for (int j = 0; j < OK; j++) {
						for (int k=t; k < windowSize+t; k++) {
							sumHold += w_a[j]*x_Test[k]  + b_a[0];
						}
					}
					a_Test[t] = tanh(sumHold);
				}	
				double sumHoldA_t;
				if (t > 0) {
					sumHoldA_t = 0.0;
					for (int j = 0; j < OK; j++) {
						for(int k = t; k < windowSize+t; k++) {
							sumHoldA_t += w_a[j]*x_Test[k]  + u_a[0]*output_Test[t-1] + b_a[0];
						}
					}
					a_Test[t] = tanh(sumHoldA_t); 
				}
			// input gate
			// i_t = g(W_i * x_t + U_i * out_t-1 + b_i)
				double sumI_t;
				if (t == 0) {
					sumI_t = 0.0;
					for (int j = 0; j < OK; j++) {
						for (int k = t; k < windowSize+t; k++) {
							sumI_t += w_i[j]*x_Test[k]+b_i[0];
						}
					}
					i_Test[t] = sigmoid_proper(sumI_t);	
				}
				double sumI_t2;
				if ( t > 0) {
					sumI_t2 = 0.0;
					for (int j = 0; j < OK; j++) {
						for (int k = t; k < windowSize+t; k++) {
							sumI_t2 += w_i[j]*x_Test[k]+u_i[0]*output_Test[t-1]+b_i[0];
						}	
					}	
					i_Test[t] = sigmoid_proper(sumI_t2);	
				}	

			// forget gate
				double sumF_t;
				if (t == 0) {
					sumF_t = 0.0;
					for (int j = 0; j < OK; j++) {
						for (int k = t; k < windowSize+t; k++) {
							sumF_t += w_f[j]*x_Test[k]+b_f[0];
						}
					}
					cout << endl;
					f_Test[t] = sigmoid_proper(sumF_t);
				}
				double sumF_t2;
				if (t > 0) {
					sumF_t2= 0.0;
					for (int j = 0; j < OK; j++) {
						for (int k = t; k < windowSize+t; k++) {
							sumF_t2 += w_f[j]*x_Test[k]+u_f[0]*output_Test[t-1]+b_f[0];
						}
					}
					f_Test[t] = sigmoid_proper(sumF_t2);
				}
		
		
		
			// output gate
				double sumO_t;
				if (t == 0) {
					sumO_t = 0.0;
					for (int j = 0; j < OK; j++) {
						for (int k = t; k< windowSize+t; k++) {
							sumO_t += w_o[j]*x_Test[k]+b_o[j];
						} 
					}
					o_Test[t] = sigmoid_proper(sumO_t);
					cout << "O_test value: "<< o_Test[t] << endl;
				}
				double sumO_t2;
				if( t > 0) {
					sumO_t2 = 0.0;
					for (int j = 0; j < OK; j++) {
						for (int k=t; k < windowSize+t; k++) {
							sumO_t2 += w_o[j]*x_Test[k]+u_o[0]*output_Test[t-1]+b_o[0];
						}
					}
					o_Test[t] = sigmoid_proper(sumO_t2);
					cout << "O_test value: "<< o_Test[t] << endl;
				}	
	                // state
				if (t == 0) {
					state_Test[t] = (i_Test[t] * a_Test[t]);
				}  
				if( t >= 1) {
					state_Test[t] = (state_Test[t-1] * f_Test[t]) + (i_Test[t] * a_Test[t]);
				}

				// output
				// use linear equation instead of tanh (in order to output actual value instead of (-1,1)
				output_Test[t] = o_Test[t]*tanh(state_Test[t]);
			}

			for (int i = 0, k=0; i < testAmount; i++) {
				if (output_Test[i] <=-0.5) {

					fprintf(predictionSheet, "%d %.4f %.4f %.4f\n",count, x_Test[i], target_Test[i], prediction_Test[i]);
					count++;
				}

			}
		}
	fclose(predictionSheet);
	cout << "Epoch count: " << trainingEpoch << endl;
	cout << "Training count: " << trainingAmount << endl;
	cout << "Rho: " << rho << endl;
	cout << "Window size: " << windowSize << endl;
	cout << "\n";
	cout << "Training: " << endl;
	printArray(windowSize, x_t, "Input sequence window ");
	printArray(1, target_t, "Target values");
	cout << "Is it -1? ";
	cout << output_t[trainingAmount-1] << endl;
	printMatrix(w, "Input (hidden) Weights");
	printMatrix(u, "Output Weights");
	cout << "Test information: " << endl;
	cout << "Test amount: " << testAmount << endl;
	printArray(windowSize, x_Test, "Input sequence window ");
	printArray(windowSize, output_Test, "output probabilities");
	printArray(trainingAmount, target_t, "Target");
	printArray(trainingAmount, output_t, "Output");
	printArray(trainingAmount, loss_t, "Error");
	delete[] a_t;
        delete[] i_t;
        delete[] f_t;
        delete[] o_t;
        delete[] deltaA_t;
        delete[] deltaI_t;
        delete[] deltaF_t;
        delete[] deltaO_t;
        delete[] output_t;
        delete[] loss_t;
	delete[] dOutput;
	delete[] output_Test;
	delete[] a_Test;
	delete[] i_Test;
	delete[] f_Test;
	delete[] o_Test;
	delete[] state_Test;
	return 0;
}
