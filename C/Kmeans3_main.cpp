// EE 569 Homework #4
// date:	Apr. 23th, 2017
// Name:	Shuo Wang
// ID:		8749390300
// email:	wang133@usc.edu

// Compiled on WINDOWS 10 with Visual C++ and Opencv 3.2
// solution for Problem 1 (c) The K-means for the first fully-connected layer
// Directly open the Kmeans3.exe to run the program

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

Mat srcF(14 * 14 * 50000, 6, CV_32FC1, Scalar(0));
Mat srcFi(50000, 5 * 5 * 16, CV_32FC1, Scalar(0));
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
		float Fil[16][150] = { 0 };
		ifstream infile;
		infile.open("D:/EE569_Assignment/4/C++/Kmeans2/x64/Debug/Weight_W.txt");//Load the weight number of the third layer
		float* ptr = &Fil[0][0];
		while (!infile.eof())
		{
			infile >> *ptr;
			ptr++;
		}
		infile.close();
		string fileName = "D:/EE569_Assignment/4/C++/Kmeans2/x64/Debug/Raw_Data.txt";//load the database for the input for second convolution layer
		LoadData(fileName, srcF, 14 * 14 * 50000, 6);

		Mat FILTER(16, 150, CV_32FC1);
		for (int n = 0; n < 16; n++)
		{
			for (int d = 0; d < 150; d++)
			{
				FILTER.at<float>(n, d) = Fil[n][d];
			}
		}
		cout << FILTER << endl;
		system("pause");
		for (int ima = 0; ima < 50000; ima++)
		{
			Mat srck(1, 150, CV_32FC1, Scalar(0));
			Mat srcc(1, 10 * 10 * 16, CV_32FC1, Scalar(0));
			Mat srccm(1, 5 * 5 * 16, CV_32FC1, Scalar(0));
			int s0 = 0;
			for (int i = 2; i < 14 - 2; i = i + 1)
			{
				for (int j = 2; j < 14 - 2; j = j + 1)
				{
					int ss = 0;
					for (int m = -2; m < 3; m++)
					{
						for (int n = -2; n < 3; n++)
						{
							srck.at<float>(0, ss) = srcF.at<float>(14 * 14 * ima + 14 * (i + m) + j + n, 0);
							srck.at<float>(0, ss + 25) = srcF.at<float>(14 * 14 * ima + 14 * (i + m) + j + n, 1);
							srck.at<float>(0, ss + 50) = srcF.at<float>(14 * 14 * ima + 14 * (i + m) + j + n, 2);
							srck.at<float>(0, ss + 75) = srcF.at<float>(14 * 14 * ima + 14 * (i + m) + j + n, 3);
							srck.at<float>(0, ss + 100) = srcF.at<float>(14 * 14 * ima + 14 * (i + m) + j + n, 4);
							srck.at<float>(0, ss + 125) = srcF.at<float>(14 * 14 * ima + 14 * (i + m) + j + n, 5);
							ss++;
						}
					}
					for (int n = 0; n < 16; n++)
					{
						float sum = 0;
						for (int k = 0; k < 150; k++)
						{
							sum = sum + srck.at<float>(0, k) * FILTER.at<float>(n, k);//calculate the output of second convolution layer
						}
						srcc.at<float>(0, 16 * s0 + n) = sum;
					}
					//cout << s0 << endl;
					s0++;
				}
			}
			for (int j = 0; j < 16; j++)
			{
				double avg = 0;
				for (int i = 0; i < 10 * 10; i++)
				{
					avg = avg + (double)srcc.at<float>(0, 16 * i + j) / 10 / 10;
				}
				for (int i = 0; i < 10 * 10; i++)
				{
					srcc.at<float>(0, 16 * i + j) = srcc.at<float>(0, 16 * i + j) - avg;
				}
				for (int i0 = 0; i0 < 5; i0++)
				{
					for (int j0 = 0; j0 < 5; j0++)
					{
						Mat srcmax(2, 2, CV_32FC1);
						srcmax.at<float>(0, 0) = srcc.at<float>(0, 16 * (10 * 2 * i0 + 2 * j0) + j);
						srcmax.at<float>(0, 1) = srcc.at<float>(0, 16 * (10 * 2 * i0 + 2 * j0 + 1) + j);
						srcmax.at<float>(1, 0) = srcc.at<float>(0, 16 * (10 * (2 * i0 + 1) + 2 * j0) + j);
						srcmax.at<float>(1, 1) = srcc.at<float>(0, 16 * (10 * (2 * i0 + 1) + 2 * j0 + 1) + j);
						double maxi, mini;
						minMaxLoc(srcmax, &mini, &maxi);
						srccm.at<float>(0, 16 * (5 * i0 + j0) + j) = maxi;//max-pooling
					}
				}
			}
			for (int i = 0; i < 5 * 5 * 16; i++)
			{
				srcFi.at<float>(ima, i) = srccm.at<float>(0, i);
			}
		}
		//cout << srcFi << endl;
		string fileName1 = "D:/EE569_Assignment/4/C++/Kmeans3/x64/Debug/Raw_Data.txt";//Save the input for the first fully-connected layer
		WriteData(fileName1, srcFi);
		system("pause");
		return 0;
	}
	else
	{
		string fileName = "D:/EE569_Assignment/4/C++/Kmeans3/x64/Debug/Raw_Data.txt";//Load the input for the first fully-connected layer
		LoadData(fileName, srcFi, 50000, 5 * 5 * 16);
	}

	Mat bestLabels(50000, 1, CV_32FC1);
	Mat center(120, 5 * 5 * 16, CV_32FC1);
	kmeans(srcFi, 120, bestLabels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 300, 1), 3, KMEANS_PP_CENTERS, center);//K-means
	cout << "The center is: " << endl;

	char *aaa = { "D:/EE569_Assignment/4/C++/Kmeans3/x64/Debug/Weight.txt" };//save the original centroid
	ofstream outfile;
	outfile.open(aaa);
	if (outfile.is_open())
	{
		for (int n = 0; n < 120; n++)
		{
			for (int d = 0; d < 5 * 5 * 16; d++)
			{
				outfile << center.at<float>(n, d) << '\t';
			}
		}
	}
	outfile.close();
	cout << format(center, Formatter::FMT_PYTHON) << endl;

	float Fil2[120][5 * 5 * 16] = { 0 };
	ifstream infile;
	infile.open("D:/EE569_Assignment/4/C++/Kmeans3/x64/Debug/Weight.txt");
	float* ptr1 = &Fil2[0][0];
	while (!infile.eof())
	{
		infile >> *ptr1;
		ptr1++;
	}
	infile.close();
	for (int n = 0; n < 120; n++)
	{
		double norm1 = 0;
		for (int i = 0; i < 5 * 5 * 16; i++)
		{
			norm1 = norm1 + abs(Fil2[n][i]);
		}
		for (int i = 0; i < 5 * 5 * 16; i++)
		{
			Fil2[n][i] = Fil2[n][i] / norm1;
		}
	}
	char *aaa1 = { "D:/EE569_Assignment/4/C++/Kmeans3/x64/Debug/Weight_W.txt" };//save the normalized centroid, which is the initialized parameter of the first layer

	//ofstream outfile;
	outfile.open(aaa1);
	if (outfile.is_open())
	{
		for (int n = 0; n < 120; n++)
		{
			for (int d = 0; d < 5 * 5 * 16; d++)
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