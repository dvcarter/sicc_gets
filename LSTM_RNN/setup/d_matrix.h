#include <vector>
using namespace std;
template <typename T>
class matrix {
	public:
		matrix(int numRows = 1, int numCols = 1, const T& initVal = T());

		std::vector<T> & operator[] (int i);


		int rows();
		int cols();

		void resize(int numRows, int numCols);

		int nRows, nCols;
	private:

		std::vector<std::vector<T> > mat;

};

template <typename T>
matrix<T>::matrix(int numRows, int numCols, const T& initVal):
        nRows(numRows), nCols(numCols),
        mat(numRows, vector<T> (numCols, initVal)){} 


template <typename T>
vector<T>& matrix<T>::operator[] (int i) {
        if (i < 0 || i >= nRows) {
        }   
        return mat[i];
}

template <typename T>
int matrix<T>::rows() {
   return nRows;
}

template <typename T>
int matrix<T>::cols() {
   return nCols;
}
