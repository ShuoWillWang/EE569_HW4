// EE 569 Homework #4
// date:	Apr. 23th, 2017
// Name:	Shuo Wang
// ID:		8749390300
// email:	wang133@usc.edu

// Compiled on WINDOWS 10 with Visual C++ and Opencv 3.2
// solution for Problem 1 (c) The K-means for the first convolution layer
// Directly open the Kmeans.exe to run the program

#include <iostream>
#include <vector>
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <Windows.h>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

using namespace std;
using namespace cv;

Mat srcF(28 * 28 * 50000, 75, CV_32FC1, Scalar(0));
int a5[6] = { 0 };

int main()
{
	for (int fol = 0; fol < 5; fol++)
	{
		int sa[10001] = { 0 };
		int summFa = 0;
		for (int i = 0; i < fol + 1; i++)
		{
			summFa = summFa + a5[i];
		}
		for (int ima = 0; ima < 10000; ima++)
		{
			char buffer[50];
			_itoa(ima, buffer, 10);
			char file_address[5][56] = { "D:/EE569_Assignment/4/C++/CIFAR/x64/Debug/data_batch_1/",  "D:/EE569_Assignment/4/C++/CIFAR/x64/Debug/data_batch_2/",  "D:/EE569_Assignment/4/C++/CIFAR/x64/Debug/data_batch_3/",  "D:/EE569_Assignment/4/C++/CIFAR/x64/Debug/data_batch_4/",  "D:/EE569_Assignment/4/C++/CIFAR/x64/Debug/data_batch_5/" };
			Mat src = imread(strcat(file_address[fol], strcat(buffer, ".jpg")));//open the image
			Mat src0(src.rows, src.cols, CV_32FC3);
			Mat src1(src.rows, src.cols, CV_32FC3);
			double avg1 = 0, avg2 = 0, avg3 = 0;
			for (int i = 0; i < src.rows; i++)
			{
				for (int j = 0; j < src.cols; j++)
				{
					avg1 = avg1 + (double)src.at<Vec3b>(i, j)[0] / 32 / 32;
					avg2 = avg2 + (double)src.at<Vec3b>(i, j)[1] / 32 / 32;
					avg3 = avg3 + (double)src.at<Vec3b>(i, j)[2] / 32 / 32;
				}
			}
			for (int i = 0; i < src.rows; i++)
			{
				for (int j = 0; j < src.cols; j++)
				{
					src0.at<Vec3f>(i, j)[0] = (double)src.at<Vec3b>(i, j)[0] - avg1;
					src0.at<Vec3f>(i, j)[1] = (double)src.at<Vec3b>(i, j)[1] - avg2;
					src0.at<Vec3f>(i, j)[2] = (double)src.at<Vec3b>(i, j)[2] - avg3;
				}
			}
			Mat srck(28 * 28, 75, CV_32FC1, Scalar(0));
			Mat srcc(1, 75, CV_32FC1, Scalar(0));
			int s0 = 0;
			for (int i = 2; i < src.rows - 2; i = i + 1)
			{
				for (int j = 2; j < src.cols - 2; j = j + 1)
				{
					int ss = 0;
					for (int m = -2; m < 3; m++)
					{
						for (int n = -2; n < 3; n++)
						{
							srcc.at<float>(0, ss) = src0.at<Vec3f>(i + m, j + n)[0];
							srcc.at<float>(0, 25 + ss) = src0.at<Vec3f>(i + m, j + n)[1];
							srcc.at<float>(0, 50 + ss) = src0.at<Vec3f>(i + m, j + n)[2];
							ss++;
						}
					}
					Mat mean(2, 2, CV_64FC1), stddev(2, 2, CV_64FC1);
					meanStdDev(srcc, mean, stddev);
					if (stddev.at<double>(0, 0) >= 30.0)//filter the patch with small std
					{
						for (int s = 0; s < 75; s++)
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
			//system("pause");
			for (int i = 0; i < s0; i++)
			{
				for (int j = 0; j < 75; j++)
				{
					srcF.at<float>(summFa + sum + i, j) = srck.at<float>(i, j);//construct the database
				}
			}
			sa[ima + 1] = s0;
			//system("pause");
			//cout << s0 << endl;
		}
		int summ = 0;
		for (int i = 1; i < 10001; i++)
		{
			summ = summ + sa[i];
		}
		a5[fol + 1] = summ;
	}
	//cout << srcF << endl;
	int summF = 0;
	for (int i = 1; i < 6; i++)
	{
		summF = summF + a5[i];
	}
	cout << summF << endl;
	//system("pause");
	system("pause");
	Mat bestLabels(summF, 1, CV_32FC1);
	Mat center(6, 75, CV_32FC1);
	kmeans(srcF, 6, bestLabels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 300, 1), 3, KMEANS_PP_CENTERS, center);//do the K-means
	system("pause");
	cout << "The center is: " << endl;
	int bgr[75] = { 0 };
	for (int i = 0; i < 25; i++)
	{
		bgr[i] = i + 50;
		bgr[i + 25] = i + 25;
		bgr[i + 50] = i + 0;
	}
	char *aaa = { "D:/EE569_Assignment/4/C++/Kmeans/x64/Debug/Weight.txt" };//save the original centroid

	ofstream outfile;
	outfile.open(aaa);
	if (outfile.is_open())
	{
		for (int n = 0; n < 6; n++)
		{
			for (int d = 0; d < 75; d++)
			{
				outfile << center.at<float>(n, bgr[d]) << '\t';
			}
		}
	}
	outfile.close();
	cout << format(center, Formatter::FMT_PYTHON) << endl;

	float Fil[6][75] = { 0 };
	ifstream infile;
	infile.open("D:/EE569_Assignment/4/C++/Kmeans/x64/Debug/Weight.txt");
	float* ptr = &Fil[0][0];
	while (!infile.eof())
	{
		infile >> *ptr;
		ptr++;
	}
	infile.close();
	
	for (int n = 0; n < 6; n++)
	{
		double norm1 = 0, norm2 = 0, norm3 = 0;
		for (int i = 0; i < 75; i++)
		{
			norm1 = norm1 + (Fil[n][i]);
			//norm2 = norm2 + abs(Fil[n][25 + i]);
			//norm3 = norm3 + abs(Fil[n][50 + i]);
		}
		for (int i = 0; i < 75; i++)
		{
			Fil[n][i] = Fil[n][i] / (norm1);
			//Fil[n][i + 25] = Fil[n][i + 25] / norm2;
			//Fil[n][i + 50] = Fil[n][i + 50] / norm3;
		}
	}
	char *aaa1 = { "D:/EE569_Assignment/4/C++/Kmeans/x64/Debug/Weight_W.txt" };//save the normalized centroid, which is the initialized parameter of the first layer

	//ofstream outfile;
	outfile.open(aaa1);
	if (outfile.is_open())
	{
		for (int n = 0; n < 6; n++)
		{
			for (int d = 0; d < 75; d++)
			{
				outfile << Fil[n][d] << '\t';
			}
		}
	}
	outfile.close();

	float Filaa[5][5][3][6] = { 0 };
	for (int n = 0; n < 6; n++)
	{
		for (int s = 0; s < 3; s++)
		{
			for (int i = 0; i < 5; i++)
			{
				for (int j = 0; j < 5; j++)
				{
					Filaa[i][j][s][n] = Fil[n][25 * s + 5 * i + j];
				}
			}
		}
	}
	char *aaa11 = { "D:/EE569_Assignment/4/C++/Kmeans/x64/Debug/Weight_Wa.txt" };

	//ofstream outfile;
	outfile.open(aaa11);
	if (outfile.is_open())
	{
		for (int i = 0; i < 5; i++)
		{
			for (int j = 0; j < 5; j++)
			{
				for (int s = 0; s < 3; s++)
				{
					for (int n = 0; n < 6; n++)
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