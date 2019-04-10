#include <fstream>
#include <sstream>
#include <vector>

using namespace std;

void parseCSV(vector<vector<double> > *inputMatrix, string fileName) {
	ifstream file;
	file.open(fileName);

	string line, value;

	while (getline(file, line)) {
		vector<double> v;
		stringstream s (line);
		while (getline (s, value, ',')) {
			v.push_back(stod (value));
		}
		inputMatrix->push_back(v);
	}

}
