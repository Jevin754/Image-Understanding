#include "myfunction.h"






int main(){

	char *img1Name = ".\\images\\z1.jpg";
	char *img2Name = ".\\images\\z2.jpg";
	char *descripName1 = ".\\data\\descriptor1.txt";
	char *descripName2 = ".\\data\\descriptor2.txt";

	Mat img1 = imread(img1Name);
	Mat img2 = imread(img2Name);
	imshow("Img1", img1);
	imshow("Img2", img2);
	Mat dst, init;
	if (img1.empty() || img2.empty()){
		cout << "error" << endl;
		waitKey(0);
		return -1;
	}

	opencvMatch(img1, img2);

	cout << endl;
	cout << "============= �Լ�ʵ�ֵ�SIFT����ƥ���� ===========" <<endl;
	clock_t start, end, t1;
	start = clock();
	vector<Keypoint> features1;
	vector<Keypoint> features2;
	getSiftFeatures(img1, features1);
	t1 = clock();
	getSiftFeatures(img2, features2);
	end = clock();
	int totalMatch = featureMatch(features1, features2, img1, img2);
	cout << "ͼ��1���������:" << features1.size() << endl;
	cout << "ͼ��2���������:" << features2.size() << endl;
	cout << "SIFT�㷨��ʱ��ͼ1���� " << (t1 - start)*1.0 / CLOCKS_PER_SEC << "s"<<endl;
	cout << "SIFT�㷨��ʱ��ͼ2���� " << (end - t1)*1.0 / CLOCKS_PER_SEC << "s"<<endl;
	cout << "SIFT�㷨��ʱ��2��ͼ���ƣ��� " << (end - start)*1.0 / CLOCKS_PER_SEC << "s"<<endl;
	cout << "ƥ������� " << totalMatch << endl;
	

	WriteFeatures(features1, descripName1);
	WriteFeatures(features2, descripName2);
	Circlekeypoints(img1, features1, "MySIFT_Img1");
	Circlekeypoints(img2, features2, "MySIFT_Img2");
	
	cout << "SIFT�������Ѿ�д��data�ļ���" << endl;
	waitKey(0);
	getchar();
	return 0;
}


