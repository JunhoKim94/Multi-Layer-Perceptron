#include <iostream>
#include <Eigen/Dense>

#include <cstdlib>
#include <ctime>

#define MAX2(a,b) if (a) > (b) ? (a) : (b)

using namespace Eigen;
using namespace std;

double sigmoid(const double& input)
{
	double epsilon = 1e-8;
	return 1 / (1 + exp(-input) + epsilon);
}

void Random_Init(MatrixXd& input)
{
	for (int i = 0; i < input.rows(); i++)
	{
		for (int j = 0; j < input.cols(); j++)
		{
			input(i, j) = (double)rand()/ RAND_MAX;
		}
	}
}

class Sigmoid
{
public:
	MatrixXd output, dev;
	
	int rows, cols;

	MatrixXd forward(const MatrixXd& input)
	{
		rows = input.rows();
		cols = input.cols();

		output = MatrixXd(rows, cols);

		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
			{
				output(i, j) = sigmoid(input(i, j));
			}

		}
		//same dim as input(N,H)
		return output;

	}
	MatrixXd backward(const MatrixXd& dout)
	{
		dev = MatrixXd(rows, cols);

		//dout * output * (1 - output);
		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
			{
				dev(i, j) = dout(i, j) * output(i, j) * (1 - output(i, j));
			}
		}
		return dev;
	}

};


class Single_Layer
{
public:
	int input, output;

	MatrixXd weight, dw;
	MatrixXd bias, db;

	Sigmoid sigmoid;


	MatrixXd x_input;

	Single_Layer(const int& a, const int& b)
	{
		input = a;
		output = b;

		weight = MatrixXd(input, output);
		bias = MatrixXd(1, output);

		dw = MatrixXd(input, output);
		db = MatrixXd(1, output);
	}

	//Initialize weight & bias by 0~1 random values
	void Initializer(const bool& init = false)
	{
		double temp;

		for (int i = 0; i < output; i++)
		{
			for (int j = 0; j < input; j++)
			{
				temp = (double)rand();
				weight(j,i) = temp / RAND_MAX;
			}
			bias(0, i) = (double)rand() / RAND_MAX;
		}
	
	}

	MatrixXd forward(const MatrixXd& x_in)
	{
		//x_in = Batch(N) x input (D)
		//weight = input(D) x output(H)
		//bias = 1 x output(H)
		x_input = x_in;
		
		int Batch;
		Batch = x_input.size() / input;

		MatrixXd out;
		
		//out = Batch(N) x output(H)
		out = x_in * weight;

		for (int i = 0; i < output; i++)
		{
			for (int j = 0; j < Batch; j++)
			{
				out(j, i) += bias(0, i);
			}
		}

		return sigmoid.forward(out);
		

	}

	MatrixXd backward(const MatrixXd& dout)
	{
		MatrixXd db(1, dout.cols());

		dw = x_input.transpose() * sigmoid.backward(dout);
		for (int i = 0; dout.rows(); i++)
		{
			db += dout.row(dout.rows());
		}

		return dout * weight.transpose(); // dx
	}

};

int main()
{
	Single_Layer layer1(5,3);

	cout << layer1.weight << layer1.bias << endl;

	
	srand((unsigned int)time(NULL));
	
	layer1.Initializer(true);

	cout << layer1.weight << layer1.bias << endl;

	MatrixXd input(5,5);

	for (int i = 0; i < 5; i++)
	{
		for (int j = 0; j < 5; j++)
		{
			input(i, j) = rand();
		}
	}
	cout << input.rows() << endl;
	cout << layer1.forward(input) << endl;

	MatrixXd dout(5, 3);

	Random_Init(dout);

	layer1.backward(dout);

	cout <<layer1.db << layer1.dw << endl;
	
	
}