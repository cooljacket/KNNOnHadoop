#include <iostream>
#include <fstream>
using namespace std;


int main() {
	fstream res("output", ios::in);
	string actual, predict;
	int total = 0, correct = 0;

	while (res >> actual >> predict) {
		++total;
		if (actual == predict)
			++correct;
	}

	printf("Accuracy: %d/%d, %.3lf\n", correct, total, correct*1.0/total);
	res.close();

	return 0;
}