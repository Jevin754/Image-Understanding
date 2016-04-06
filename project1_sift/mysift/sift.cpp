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
	cout << "============= 自己实现的SIFT算子匹配结果 ===========" <<endl;
	clock_t start, end, t1;
	start = clock();
	vector<Keypoint> features1;
	vector<Keypoint> features2;
	getSiftFeatures(img1, features1);
	t1 = clock();
	getSiftFeatures(img2, features2);
	end = clock();
	int totalMatch = featureMatch(features1, features2, img1, img2);
	cout << "图像1特征点个数:" << features1.size() << endl;
	cout << "图像2特征点个数:" << features2.size() << endl;
	cout << "SIFT算法用时（图1）： " << (t1 - start)*1.0 / CLOCKS_PER_SEC << "s"<<endl;
	cout << "SIFT算法用时（图2）： " << (end - t1)*1.0 / CLOCKS_PER_SEC << "s"<<endl;
	cout << "SIFT算法用时（2张图共计）： " << (end - start)*1.0 / CLOCKS_PER_SEC << "s"<<endl;
	cout << "匹配点数： " << totalMatch << endl;
	

	WriteFeatures(features1, descripName1);
	WriteFeatures(features2, descripName2);
	Circlekeypoints(img1, features1, "MySIFT_Img1");
	Circlekeypoints(img2, features2, "MySIFT_Img2");
	
	cout << "SIFT描述子已经写入data文件夹" << endl;
	waitKey(0);
	getchar();
	return 0;
}


