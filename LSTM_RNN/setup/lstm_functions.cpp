// activation functions
// (sigmoid) binary
void sigmoid_vec(int n, float net[], float y[]) {
        int i;
        for (i = 0; i < n; i++) {
                y[i] = 1/(1+exp(-net[i]));
        }   
}

float sigmoid_proper(float x) {
	float y;
	y = 1/(1+exp(-x));
	return y;
}
void sigmoidGateActivation(int gateSize, int weightSize, float * gate, float * hiddenInputWeight, float * input, float * outputWeight, float * predOutput, float * bias, string gateLabel) {
	float * weight_sum = new float[gateSize];
	initializeArray(gateSize, weight_sum, 0.0);
	for (int i = 0; i < gateSize; i++) {
//                cout << gateLabel << " level " << i << endl;
                for (int j = 0; j < weightSize; j++) {
                        weight_sum[i] =  hiddenInputWeight[j]*input[i]+outputWeight[j]*predOutput[i]+bias[j];
                }
                sigmoid_vec(gateSize, weight_sum, gate);
                 gate[i] += gate[i];
//                 printArray(gateSize, bias, gateLabel+ " bias");
//                 printArray(gateSize, gate, gateLabel);
        }
	delete[] weight_sum;
}

// tanh 
void tanhActivation(int gateSize, int weightSize, float * gate, float * hiddenInputWeight, float * input, float * outputWeight, float * predOutput, float * bias, string gateLabel, string gateShortName) {
     for (int i = 0; i < gateSize; i++) {
                for (int j = 0; j < weightSize; j++) {
                        gate[i] += tanh(hiddenInputWeight[j]*input[i]  + outputWeight[j]*predOutput[i] + bias[j]);
                }
	}
}
// soft-max


// forward propagation
void gateSetup(int gateSize, int weightSize, float * gate, float * hiddenInputWeight , float * outputWeight, float * bias, float *weight_sum, float * outputAt, string gateLabel) {
        initializeArray(gateSize, gate, 0.0);
        initializeArray(weightSize, hiddenInputWeight, 0.0);
	initializeArray(weightSize, outputWeight, 0.0);
        initializeArray(weightSize, bias, 0.0);
        initializeArray(weightSize, weight_sum, 0.0);

//	cout << "Initial " << gateLabel <<" values " << endl;
//        printArray(gateSize, gate, gateLabel);
//        printArray(weightSize, bias, gateLabel+" bias");
//        printArray(weightSize, hiddenInputWeight, gateLabel+" hidden weight * input_t");
//        printArray(weightSize, outputWeight, gateLabel+ " output weight * output_t-1");
//        printArray(gateSize,outputAt , "mock prediction");	
}

// input activation 


// input gate

// forget gate

// output gate

//

// backpropagation
// 8 equations for updating 8 weights
//
// W_a, W_i, W_f, W_o
//
// U_a, U_i, U_f, U_o


                        // W_t = [W_a_t, W_i_t, W_f_t, W_o_t]
                        // gates_t = [a_t, i_t, f_t, o_t]

                        // d(gates_t) = [d(a_t), d(i_t), d(f_t), d(o_t)]

                        // change in x_t
                        // d(x_t) = W_t * d(gates_t)


                        // change in W_t
                        // d(W_t) = sum (d(gates_t) * x_t)

                        // change in U_t
                        // d(U_t) = sum (d(gates_t+1) *  out_t)

                        // change in b_t
                        // d(b_t) = d(gates_t+1)
