// EE 569 Homework #4
// date:	Apr. 23th, 2017
// Name:	Shuo Wang
// ID:		8749390300
// email:	wang133@usc.edu

// Compiled on WINDOWS 10 with Visual C++ and Opencv 3.2
// solution for Problem 1 (c) Acquire the 32x32 images of the CIFAR-10 from the binary package (avaliaboe for cifar-10-batches-binary)
// Directly open the CIFAR.exe to run the program

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

void randge(int *ind, int classi, int num)//random number generator
{
	srand((unsigned)time(NULL));
	for (int i = 0; i < classi; i++)
	{
		ind[i] = rand() % num;
		for (int j = 0; j < i; j++)
		{
			if (ind[i] == ind[j])
			{
				ind[i] = rand() % num;
				j = 0;
			}
		}
	}
	for (int i = 0; i < classi; i++)
	{
		cout << ind[i] << endl;
	}
}

void read_cifar_bin(string file_address, vector<Mat>& image, vector<int>& flag)
{
	int width = 32, height = 32;
	ifstream fin(file_address, ios::binary);
	while (!fin.eof())
	{
		char flag_tmp;
		unsigned char tmp;
		Mat image_tmp(width, height, CV_8UC3);
		fin.read((char *)&flag_tmp, sizeof(flag_tmp));

		for (int j = 2; j >= 0; j--)
		{
			for (int r = 0; r < image_tmp.rows; r++)
				for (int c = 0; c < image_tmp.cols; c++)
				{
					fin.read((char *)&tmp, sizeof(tmp));
					image_tmp.at<Vec3b>(r, c)[j] = tmp;
				}
		}
		image.push_back(image_tmp);
		flag.push_back(flag_tmp);
	}
}

void write_cifar_bin(string file_address, vector<string>& image_address, vector<int>& flag)
{
	ofstream fout(file_address, ios::binary);
	for (size_t i = 0; i < image_address.size(); i++)
	{
		Mat image_tmp = imread(image_address[i], 1);
		resize(image_tmp, image_tmp, Size(32, 32));


		int pix[1024];
		char flag_tmp = flag[i];
		fout.write((char *)&flag_tmp, sizeof(flag_tmp));

		for (int j = 2; j >= 0; j--)
		{
			for (int r = 0; r < image_tmp.rows; r++)
				for (int c = 0; c < image_tmp.cols; c++)
				{
					unsigned char tmp = image_tmp.at<Vec3b>(r, c)[j];
					fout.write((char *)&tmp, sizeof(tmp));
				}
		}
	}
}

int main()
{
	int question;
	cout << "Input the question you want to solve: 1) write image 2) random number: ";
	cin >> question;
	if (question == 1)
	{
		string file_address = "data_batch_5.bin";
		vector<Mat> image;
		vector<int>flag;
		read_cifar_bin(file_address, image, flag);
		ofstream mydata_batch_5("mydata_batch_5.txt");
		for (int i = 0; i < image.size(); i++)
		{
			char buffer[50];
			char address[100] = "D:/EE569_Assignment/4/C++/CIFAR/x64/Debug/data_batch_5/";
			_itoa(i, buffer, 10);
			imwrite(strcat(address, strcat(buffer, ".jpg")), image[i]);//
			mydata_batch_5 << address << " " << flag[i] << endl;//<< ".jpg"<< buffer
			//imwrite(address, image[i]);//
			cout << i << endl;
			waitKey(1);
		}
		system("pause");
		return 0;
	}
	else
	{
		double ran[5 * 5 * 3 * 9] = { 0 };
		int ran1[5 * 5 * 3 * 9] = { 0 };
		//randge(ran1, 5 * 5 * 3 * 9, 10);
		//for (int i = 0; i < 5 * 5 * 3 * 9; i++)
		//{
		//	ran[i] = (double)ran1[i] / 10;
		//	cout << ran[i] << endl;
		//}
		ofstream mydata_batch_5;
		char *aaa = { "random.txt" };//save the 12 25-D vectors in the txt to do the PCA in MATLAB
		mydata_batch_5.open(aaa);
		if (mydata_batch_5.is_open())
		{
			for (int i = 0; i < 5 * 5 * 3 * 9; i++)
			{
				mydata_batch_5 << ran[i] + 1 << endl;
			}
		}
		mydata_batch_5.close();
		system("pause");
		return 0;
	}
}