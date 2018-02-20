// EE 569 Homework #4
// date:	Apr. 23th, 2017
// Name:	Shuo Wang
// ID:		8749390300
// email:	wang133@usc.edu

// Compiled on WINDOWS 10 with Visual C++ and Opencv 3.2
// solution for Problem 1 (c) The K-means for the second fully-connected layer
// Directly open the Kmeans4.exe to run the program

#include <iostream>
#include <vector>
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <Windows.h>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iterator>

using namespace std;
using namespace cv;

Mat srcF(50000, 5 * 5 * 16, CV_32FC1, Scalar(0));
Mat srcFi(50000, 120, CV_32FC1, Scalar(0));
int a5[6] = { 0 };

int WriteData(string fileName, cv::Mat& matData)
{
	int retVal = 0;
	ofstream outFile(fileName.c_str(), ios_base::out);
	if (!outFile.is_open())
	{
		cout << "Fail to open. " << endl;
		retVal = -1;
		return (retVal);
	}
	if (matData.empty())
	{
		cout << "Matrix is NULL" << endl;
		retVal = 1;
		return (retVal);
	}

	for (int r = 0; r < matData.rows; r++)
	{
		for (int c = 0; c < matData.cols; c++)
		{
			float data = matData.at<float>(r, c);
			outFile << data << "\t";
		}
		outFile << endl;
	}

	return (retVal);
}


int LoadData(string fileName, cv::Mat& matData, int matRows = 0, int matCols = 0, int matChns = 0)
{
	int retVal = 0;


	ifstream inFile(fileName.c_str(), ios_base::in);
	if (!inFile.is_open())
	{
		cout << "Read Faliure. " << endl;
		retVal = -1;
		return (retVal);
	}


	istream_iterator<float> begin(inFile);
	istream_iterator<float> end;
	vector<float> inData(begin, end);
	cv::Mat tmpMat = cv::Mat(inData);


	size_t dataLength = inData.size();
	if (matChns == 0)//channel number
	{
		matChns = 1;
	}
	if (matRows != 0 && matCols == 0)
	{
		matCols = dataLength / matChns / matRows;
	}
	else if (matCols != 0 && matRows == 0)
	{
		matRows = dataLength / matChns / matCols;
	}
	else if (matCols == 0 && matRows == 0)
	{
		matRows = dataLength / matChns;
		matCols = 1;
	}
	if (dataLength != (matRows * matCols * matChns))
	{
		cout << "Default Output" << endl;
		retVal = 1;
		matChns = 1;
		matRows = dataLength;
	}


	matData = tmpMat.reshape(matChns, matRows).clone();

	return (retVal);
}

int main()
{
	int type;
	cout << "What thing do you want to do: 1) Save the data; 2) Load the data and compute: ";
	cin >> type;
	if (type == 1)
	{
		float Fil[120][5 * 5 * 16] = { 0 };
		ifstream infile;
		infile.open("D:/EE569_Assignment/4/C++/Kmeans3/x64/Debug/Weight_W.txt");//Load the weight number of the fourth layer
		float* ptr = &Fil[0][0];
		while (!infile.eof())
		{
			infile >> *ptr;
			ptr++;
		}
		infile.close();
		string fileName = "D:/EE569_Assignment/4/C++/Kmeans3/x64/Debug/Raw_Data.txt";//load the database for the input for first fully-connected layer
		LoadData(fileName, srcF, 50000, 5 * 5 * 16);

		Mat FILTER(120, 5 * 5 * 16, CV_32FC1);
		for (int n = 0; n < 120; n++)
		{
			for (int d = 0; d < 5 * 5 * 16; d++)
			{
				FILTER.at<float>(n, d) = Fil[n][d];
			}
		}
		//cout << FILTER << endl;
		
		for (int ima = 0; ima < 50000; ima++)
		{
			Mat srck(1, 5 * 5 * 16, CV_32FC1, Scalar(0));
			Mat srcc(1, 120, CV_32FC1, Scalar(0));
			for (int ss = 0; ss < 5 * 5 * 16; ss++)
			{
				srck.at<float>(0, ss) = srcF.at<float>(ima, ss);
			}
			
			for (int s0 = 0; s0 < 120; s0++)
			{
				float sum = 0;
				for (int ss = 0; ss < 5 * 5 * 16; ss++)
				{
					sum = sum + srck.at<float>(0, ss) *  FILTER.at<float>(s0, ss);//calcutate the output of the first fully-connected
				}
				srcc.at<float>(0, s0) = sum;
			}
			float avg = 0;
			for (int s0 = 0; s0 < 120; s0++)
			{
				avg = avg + (double)srcc.at<float>(0, s0) / 120;
			}
			for (int s0 = 0; s0 < 120; s0++)
			{
				srcc.at<float>(0, s0) = srcc.at<float>(0, s0) - avg;
				srcFi.at<float>(ima, s0) = srcc.at<float>(0, s0);
			}
		}
		string fileName1 = "D:/EE569_Assignment/4/C++/Kmeans4/x64/Debug/Raw_Data.txt";//save the input data for the second fully-connected layer
		WriteData(fileName1, srcFi);
		system("pause");
		return 0;
	}
	else
	{
		string fileName = "D:/EE569_Assignment/4/C++/Kmeans4/x64/Debug/Raw_Data.txt";//load the input data for the second fully-connected layer
		LoadData(fileName, srcFi, 50000, 120);
	}

	Mat bestLabels(50000, 1, CV_32FC1);
	Mat center(84, 120, CV_32FC1);
	kmeans(srcFi, 84, bestLabels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 300, 1), 3, KMEANS_PP_CENTERS, center);//Kmeans
	cout << "The center is: " << endl;

	char *aaa = { "D:/EE569_Assignment/4/C++/Kmeans4/x64/Debug/Weight.txt" };//save the original centroid
	ofstream outfile;
	outfile.open(aaa);
	if (outfile.is_open())
	{
		for (int n = 0; n < 84; n++)
		{
			for (int d = 0; d < 120; d++)
			{
				outfile << center.at<float>(n, d) << '\t';
			}
		}
	}
	outfile.close();
	cout << format(center, Formatter::FMT_PYTHON) << endl;

	float Fil2[84][120] = { 0 };
	ifstream infile;
	infile.open("D:/EE569_Assignment/4/C++/Kmeans4/x64/Debug/Weight.txt");
	float* ptr1 = &Fil2[0][0];
	while (!infile.eof())
	{
		infile >> *ptr1;
		ptr1++;
	}
	infile.close();
	for (int n = 0; n < 84; n++)
	{
		double norm1 = 0;
		for (int i = 0; i < 120; i++)
		{
			norm1 = norm1 + (Fil2[n][i]);
		}
		for (int i = 0; i < 120; i++)
		{
			Fil2[n][i] = Fil2[n][i] / norm1;
		}
	}
	char *aaa1 = { "D:/EE569_Assignment/4/C++/Kmeans4/x64/Debug/Weight_W.txt" };//save the normalized centroid, which is the initialized parameter of the first layer
	outfile.open(aaa1);
	if (outfile.is_open())
	{
		for (int n = 0; n < 84; n++)
		{
			for (int d = 0; d < 120; d++)
			{
				outfile << Fil2[n][d] << '\t';
			}
		}
	}
	outfile.close();
	system("pause");
	system("pause");
	system("pause");
}