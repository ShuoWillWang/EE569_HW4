// EE 569 Homework #4
// date:	Apr. 23th, 2017
// Name:	Shuo Wang
// ID:		8749390300
// email:	wang133@usc.edu

// Compiled on WINDOWS 10 with Visual C++ and Opencv 3.2
// solution for Problem 1 (c) The K-means for the second convolution layer
// Directly open the Kmeans2.exe to run the program

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
Mat srcFi(10 * 10 * 50000, 150, CV_32FC1, Scalar(0));
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
		float Fil[6][75] = { 0 };
		ifstream infile;
		infile.open("D:/EE569_Assignment/4/C++/Kmeans/x64/Debug/Weight_W.txt");//Load the weight number of the first layer
		float* ptr = &Fil[0][0];
		while (!infile.eof())
		{
			infile >> *ptr;
			ptr++;
		}
		infile.close();

		Mat FILTER(6, 75, CV_32FC1);
		int bgr[75] = { 0 };
		for (int i = 0; i < 25; i++)
		{
			bgr[i] = i + 50;
			bgr[i + 25] = i + 25;
			bgr[i + 50] = i + 0;
		}
		for (int n = 0; n < 6; n++)
		{
			for (int d = 0; d < 75; d++)
			{
				FILTER.at<float>(n, d) = Fil[n][bgr[d]];
			}
		}
		//cout << FILTER << endl;
		for (int fol = 0; fol < 5; fol++)
		{
			for (int ima = 0; ima < 10000; ima = ima + 1)
			{
				char buffer[50];
				_itoa(ima, buffer, 10);
				char file_address[5][56] = { "D:/EE569_Assignment/4/C++/CIFAR/x64/Debug/data_batch_1/",  "D:/EE569_Assignment/4/C++/CIFAR/x64/Debug/data_batch_2/",  "D:/EE569_Assignment/4/C++/CIFAR/x64/Debug/data_batch_3/",  "D:/EE569_Assignment/4/C++/CIFAR/x64/Debug/data_batch_4/",  "D:/EE569_Assignment/4/C++/CIFAR/x64/Debug/data_batch_5/" };
				Mat src = imread(strcat(file_address[fol], strcat(buffer, ".jpg")));//open the image
				Mat srck(1, 75, CV_32FC1, Scalar(0));
				Mat srcc((src.rows - 4) * (src.cols - 4), 6, CV_32FC1, Scalar(0));
				Mat srccm(14 * 14, 6, CV_32FC1, Scalar(0));
				int s0 = 0;
				for (int i = 2; i < src.rows - 2; i++)
				{
					for (int j = 2; j < src.cols - 2; j++)
					{
						int ss = 0;
						for (int m = -2; m < 3; m++)
						{
							for (int n = -2; n < 3; n++)
							{
								srck.at<float>(0, ss) = (float)src.at<Vec3b>(i + m, j + n)[0];
								srck.at<float>(0, 25 + ss) = (float)src.at<Vec3b>(i + m, j + n)[1];
								srck.at<float>(0, 50 + ss) = (float)src.at<Vec3b>(i + m, j + n)[2];
								ss++;
							}
						}
						for (int n = 0; n < 6; n++)
						{
							float sum = 0;
							for (int k = 0; k < 75; k++)
							{
								sum = sum + srck.at<float>(0, k) * FILTER.at<float>(n, k);//calculate the initial output of first layer
							}
							srcc.at<float>(s0, n) = sum;
						}
						s0++;
					}
				}
				for (int j = 0; j < 6; j++)
				{
					for (int i0 = 0; i0 < 14; i0++)
					{
						for (int j0 = 0; j0 < 14; j0++)
						{
							Mat srcmax(2, 2, CV_32FC1);
							srcmax.at<float>(0, 0) = srcc.at<float>(28 * 2 * i0 + 2 * j0, j);
							srcmax.at<float>(0, 1) = srcc.at<float>(28 * 2 * i0 + 2 * j0 + 1, j);
							srcmax.at<float>(1, 0) = srcc.at<float>(28 * (2 * i0 + 1) + 2 * j0, j);
							srcmax.at<float>(1, 1) = srcc.at<float>(28 * (2 * i0 + 1) + 2 * j0 + 1, j);
							double maxi, mini;
							minMaxLoc(srcmax, &mini, &maxi);
							srccm.at<float>(14 * i0 + j0, j) = maxi;//max-pooling
						}
					}
					Mat mean(2, 2, CV_64FC1), stddev(2, 2, CV_64FC1);
					Mat srccmi(14 * 14, 1, CV_32FC1, Scalar(0));
					for (int i = 0; i < 14 * 14; i++)
					{
						srccmi.at<float>(i, 0) = srccm.at<float>(i, j);
					}
					meanStdDev(srccmi, mean, stddev);
					for (int i = 0; i < 14 * 14; i++)
					{
						srccm.at<float>(i, j) = srccm.at<float>(i, j) - mean.at<double>(0, 0);
					}
				}
				//cout << srccm << endl;
				//system("pause");
				for (int i = 0; i < 14 * 14; i++)
				{
					for (int j = 0; j < 6; j++)
					{
						srcF.at<float>(14 * 14 * (fol * 10000 + (int)ima) + i, j) = srccm.at<float>(i, j);
					}
				}
			}
		}

		string fileName = "D:/EE569_Assignment/4/C++/Kmeans2/x64/Debug/Raw_Data.txt";//save the input data of the second convolution layer
		WriteData(fileName, srcF);
		system("pause");
		return 0;
	}
	else
	{
		string fileName = "D:/EE569_Assignment/4/C++/Kmeans2/x64/Debug/Raw_Data.txt";
		LoadData(fileName, srcF, 14 * 14 * 50000, 6);
		//cout << srcF << endl;
	}
	int sa[50001] = { 0 };
	for (int ima = 0; ima < 50000; ima++)
	{
		Mat srck(10 * 10, 150, CV_32FC1, Scalar(0));
		Mat srcc(1, 150, CV_32FC1, Scalar(0));
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
						srcc.at<float>(0, ss) = srcF.at<float>(14 * 14 * ima + 14 * (i + m) + j + n, 0);
						srcc.at<float>(0, ss + 25) = srcF.at<float>(14 * 14 * ima + 14 * (i + m) + j + n, 1);
						srcc.at<float>(0, ss + 50) = srcF.at<float>(14 * 14 * ima + 14 * (i + m) + j + n, 2);
						srcc.at<float>(0, ss + 75) = srcF.at<float>(14 * 14 * ima + 14 * (i + m) + j + n, 3);
						srcc.at<float>(0, ss + 100) = srcF.at<float>(14 * 14 * ima + 14 * (i + m) + j + n, 4);
						srcc.at<float>(0, ss + 125) = srcF.at<float>(14 * 14 * ima + 14 * (i + m) + j + n, 5);
						ss++;
					}
				}
				Mat mean(2, 2, CV_64FC1), stddev(2, 2, CV_64FC1);
				meanStdDev(srcc, mean, stddev);
				if (stddev.at<double>(0, 0) >= 30.0)//filter the patch with small std
				{
					for (int s = 0; s < 150; s++)
					{
						srck.at<float>(s0, s) = srcc.at<float>(0, s);
					}
					s0++;
				}
			}
		}
		int sum = 0;
		for (int i = 0; i < ima + 1; i++)
		{
			sum = sum + sa[i];
		}
		for (int i = 0; i < s0; i++)
		{
			for (int j = 0; j < 150; j++)
			{
				srcFi.at<float>(sum + i, j) = srck.at<float>(i, j);
			}
		}
		sa[ima + 1] = s0;
		//cout << s0 << endl;
		//system("pause");
	}
	int summF = 0;
	for (int i = 1; i < 50001; i++)
	{
		summF = summF + sa[i];
	}
	//cout << summF << endl;
	//system("pause");
	int sumFF = (int)summF / 1;
	//system("pause");
	Mat bestLabels(summF, 1, CV_32FC1);
	Mat center(16, 150, CV_32FC1);
	kmeans(srcFi, 16, bestLabels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 300, 1), 3, KMEANS_PP_CENTERS, center);//do the K-means
	cout << "The center is: " << endl;

	char *aaa = { "D:/EE569_Assignment/4/C++/Kmeans2/x64/Debug/Weight.txt" };//save the original centroid
	ofstream outfile;
	outfile.open(aaa);
	if (outfile.is_open())
	{
		for (int n = 0; n < 16; n++)
		{
			for (int d = 0; d < 150; d++)
			{
				outfile << center.at<float>(n, d) << '\t';
			}
		}
	}
	outfile.close();
	cout << format(center, Formatter::FMT_PYTHON) << endl;

	float Fil2[16][150] = { 0 };
	ifstream infile;
	infile.open("D:/EE569_Assignment/4/C++/Kmeans2/x64/Debug/Weight.txt");
	float* ptr1 = &Fil2[0][0];
	while (!infile.eof())
	{
		infile >> *ptr1;
		ptr1++;
	}
	infile.close();
	
	for (int n = 0; n < 16; n++)
	{
		double norm1 = 0, norm2 = 0, norm3 = 0, norm4 = 0, norm5 = 0, norm6 = 0;
		for (int i = 0; i < 150; i++)
		{
			norm1 = norm1 + (Fil2[n][i]);
			//norm2 = norm2 + abs(Fil2[n][25 + i]);
			//norm3 = norm3 + abs(Fil2[n][50 + i]);
			//norm4 = norm4 + abs(Fil2[n][75 + i]);
			//norm5 = norm5 + abs(Fil2[n][100 + i]);
			//norm6 = norm6 + abs(Fil2[n][125 + i]);
		}
		for (int i = 0; i < 150; i++)
		{
			Fil2[n][i] = Fil2[n][i] / norm1;
			//Fil2[n][25 + i] = Fil2[n][25 + i] / norm2;
			//Fil2[n][50 + i] = Fil2[n][50 + i] / norm3;
			//Fil2[n][75 + i] = Fil2[n][75 + i] / norm4;
			//Fil2[n][100 + i] = Fil2[n][100 + i] / norm5;
			//Fil2[n][125 + i] = Fil2[n][125 + i] / norm6;
		}
	}
	char *aaa1 = { "D:/EE569_Assignment/4/C++/Kmeans2/x64/Debug/Weight_W.txt" };//save the normalized centroid, which is the initialized parameter of the first layer

	//ofstream outfile;
	outfile.open(aaa1);
	if (outfile.is_open())
	{
		for (int n = 0; n < 16; n++)
		{
			for (int d = 0; d < 150; d++)
			{
				outfile << Fil2[n][d] << '\t';
			}
		}
	}
	outfile.close();

	float Filaa[5][5][6][16] = { 0 };
	for (int n = 0; n < 16; n++)
	{
		for (int s = 0; s < 6; s++)
		{
			for (int i = 0; i < 5; i++)
			{
				for (int j = 0; j < 5; j++)
				{
					Filaa[i][j][s][n] = Fil2[n][25 * s + 5 * i + j];
				}
			}
		}
	}
	char *aaa11 = { "D:/EE569_Assignment/4/C++/Kmeans2/x64/Debug/Weight_Wa.txt" };

	outfile.open(aaa11);
	if (outfile.is_open())
	{
		for (int i = 0; i < 5; i++)
		{
			for (int j = 0; j < 5; j++)
			{
				for (int s = 0; s < 6; s++)
				{
					for (int n = 0; n < 16; n++)
					{
						outfile << Filaa[i][j][s][n] << '\t';
					}
				}
			}
		}
	}
	outfile.close();

	system("pause");

	system("pause");
}