#include "d_matrix.h"
#include <iomanip>

// initialize array with specific value
void initializeArray(int n, float *arr, float initialValue) {
	for (int i = 0; i < n; i++) {
		arr[i] = initialValue;
	}
}
// print array
void printArray(int n, float arr[], string str) {
        cout << "array name: " << str << endl;
        for (int i = 0; i < n; i++) {
                cout << setprecision(4) << fixed<< arr[i] << "\n";
        }   
        cout << endl;
}
void printArray2(int min, int max, float arr[], string str) {
        cout << "array name: " << str << endl;
        for (int i = min; i < min+max; i++) {
                cout << setprecision(4) << fixed<< arr[i] << "\n";
        }
        cout << endl;
}
// print matrix
void printMatrix(matrix <float> A, string label) {
        cout << "\nmatrix name: " << label << endl;
        cout << "dimensions: " << A.rows() << " by " << A.cols() << endl;
        cout << "--------------------------------------------" << endl;
        for (int i = 0; i < A.rows(); i++) {
                for (int j = 0; j < A.cols(); j++) {
                        cout << " " << setprecision(4) << fixed << (float)(A[i][j]) << " ";
                }   
               cout << endl;    
        }   
     cout << endl;    
}
void printMatrix2(int rows, int cols, matrix <float> A, string label) {
        cout << "\nmatrix name: " << label << endl;
        cout << "dimensions: " << rows << " by " << cols << endl;
        cout << "--------------------------------------------" << endl;
        for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                        cout << " " << setprecision(4) << fixed << (float)(A[i][j]) << " ";
                }  
               cout << endl;   
        }  
     cout << endl;   
}
// random array
void setupRandArray(int n, float array[], float min, float max) {
        for (int i = 0; i < n; i++) {
                array[i] = min + ((float)rand())/((float)(RAND_MAX/(max - min)));
        }
}



// input (hidden) weights
void weightInitializerInput(int rows, int cols, matrix <float> *aWeight) {
        float r = sqrtf(6.0/(float)(rows+cols));
        for (int i = 0; i < rows; i++) {
                for (int j = 0;  j < cols; j++) {
                        (*aWeight)[i][j] = -r + ((float)rand()) / ((float)RAND_MAX /(2.0 * r));
                }   
        }   

}
// prediction (output) weights
void weightInitializerOutput(int rows, int cols, matrix <float> *yWeight) {
        float r = 4.0 * sqrtf(6.0/(float)(rows+cols));
        for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                        (*yWeight)[i][j] = -r + ((float)rand())/((float) RAND_MAX/(2.0 * r)); 
                }   
        }   
}
// matrix multiplication
void matrixVectorMultiply( int rows, int cols, matrix<float> *mtrx, float x[], float c[]) {
	int i, j;
        float sum;

        for (i = 0; i < rows; i++) {
                sum = 0;
                for (j = 0; j < cols; j++) {
                        sum +=  (*mtrx)[i][j] * x[j];
                }   
                c[i] = sum;
        }   
}
// matrix to array expansion
void matrixToArray(matrix<float> mtrx, float a[], float b[], float c[], float d[]) {
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < mtrx.cols(); j++) {
		        if (i == 0) {
                              a[j] =  mtrx[i][j];
                        } else if ( i == 1) {
                                b[j] = mtrx[i][j];
                        } else if ( i == 2) {
                                c[j] = mtrx[i][j];
                        } else {
                                d[j] = mtrx[i][j];
                        }   
                }   
        }   
}
// arrays to matrix contraction
void buildMatrix(int cols, matrix<float> *mtrx, float a[], float b[], float c[], float d[]) {
	// there are only 4 gates
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < cols; j++) {
			if (i == 0) {
				(*mtrx)[i][j] = a[j];
			} else if ( i == 1) {
				(*mtrx)[i][j] = b[j];
			} else if ( i == 2) {
				(*mtrx)[i][j] = c[j];
			} else {
				(*mtrx)[i][j] = d[j];
			}
		}
	}
}
// sum elements in matrix 
void reduceMatrix(matrix<float> mtrx, matrix<float> *a) {
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < mtrx.cols(); j++) {
			if (i == 0) {
				(*a)[i][0] += mtrx[i][j];
                        } else if ( i == 1) {
                               (* a)[i][0] += mtrx[i][j];
                        } else if ( i == 2) {
                                (*a)[i][0] += mtrx[i][j];
                        } else {
                                (*a)[i][0] += mtrx[i][j];
                        }   
                }   
        } 
}
// outer product 
void outerProduct(int cols, matrix<float> mtrx, float a[], matrix<float> *outputMatrix) {
        // there are only 4 gates
        for (int i = 0; i < 4; i++) {
                for (int j = 0; j < cols; j++) {
                        if (i == 0) {
                                (*outputMatrix)[i][j] = mtrx[i][j] * a[j];
                        } else if ( i == 1) {
				(*outputMatrix)[i][j] = mtrx[i][j] * a[j];
                        } else if ( i == 2) {
				(*outputMatrix)[i][j] = mtrx[i][j] * a[j];
                        } else {
				(*outputMatrix)[i][j] = mtrx[i][j] * a[j];
                        }
                }
        }
}
// multiple matrix elements by scalar
void matrixByScalar(int cols, matrix<float> * mtrx, float scalar) {
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < cols; j++) {
			(*mtrx)[i][j] = (*mtrx)[i][j] * scalar;
		}
	}
}
// do I need addition?
void matrixSubtraction(matrix<float> * mtrx, matrix<float> otherMtrx) {
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < otherMtrx.cols(); j++) {
			(*mtrx)[i][j] = (*mtrx)[i][j] - otherMtrx[i][j];
		}
	}
}
// transpose
void transposeMatrix(matrix<float> mtrx, matrix<float> *otherMatrix) {
	for (int i = 0; i < mtrx.rows();i++) {
		for (int j = 0; j < mtrx.cols(); j++) {
			(*otherMatrix)[j][i] = mtrx[i][j];
		}
	}
}
void transposeMatrix2(int rows, int cols, matrix<float> mtrx, matrix<float> *otherMatrix) {
        for (int i = 0; i < rows;i++) {
                for (int j = 0; j < cols; j++) {
                        (*otherMatrix)[j][i] = mtrx[i][j];
                }   
        }   
}
// to one array
void matrixToArraySimplified( matrix<float> mtrx, float a[]) {
	for (int i = 0; i < mtrx.rows(); i++) {
			a[i] = mtrx[i][0];
	}
}
// to one array
void matrixToArraySimplified2(int rows, matrix<float> mtrx, float a[]) {
        for (int i = 0; i < rows; i++) {
                        a[i] = mtrx[i][0];
        }
}
