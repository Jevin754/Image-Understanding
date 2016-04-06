#ifndef HEAD_H
#define HEAD_H


//#include "myhead.h"
#include <algorithm>
#include "opencv2/opencv.hpp"
//#include <cxcore.h>
//#include "cv.h"
//#include "highgui.h"
#include "math.h"
#include "iomanip"
#include "fstream"
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include<opencv2/legacy/legacy.hpp>
#include <opencv2/features2d/features2d.hpp>
#include<opencv2/nonfree/nonfree.hpp>


#define PI 3.141592654
#define TopLevel 4
#define sigma0 1.6
#define sigma_pic0 0.5
#define S 3//numbers of extreme points
#define Feature_Size 128
#define Thresh 0.03
#define R 10
#define ORI_HIST_BINS 36
#define ORI_SIGMA_TIMES 1.5
#define ORI_WINDOW_RADIUS 3.0 * ORI_SIGMA_TIMES 
#define ORI_SMOOTH_TIMES 2
#define ORI_PEAK_RATIO 0.8
#define DESCR_MAG_THR 0.2
#define INT_DESCR_FCTR 512.0
#define MATCH_THR 0.6


struct Keypoint{
	int octave, interval;
	int x, y;
	double mag, ori;
	double scale, octave_scale;
	double offset_x, offset_y, offset_interval;
	double dx, dy;
	int descr_length;
	double descriptor[Feature_Size];
};

using namespace std;
using namespace cv;

void opencvMatch(Mat &img1, Mat &img2);

double*** CalculateDescrHist(Mat& gauss, int x, int y, double octave_scale, double ori, int bins, int width);
void InterpHistEntry(double ***hist, double xbin, double ybin, double obin, double mag, int bins, int d);
void HistToDescriptor(double ***hist, int width, int bins, Keypoint& feature);
void NormalizeDescr(Keypoint& feat);


void OrientationAssignment(vector<Keypoint>& extrema, vector<Keypoint>& features, vector<Keypoint>& gauss_pyr);
double* CalculateOrientationHistogram(Mat& gauss, int x, int y, int bins, int radius, double sigma);
bool CalcGradMagOri(Mat& gauss, int x, int y, double& mag, double& ori);
void GaussSmoothOriHist(double *hist, int n);
double DominantDirection(double *hist, int n);
void CopyKeypoint(const Keypoint& src, Keypoint& dst);
void CalcOriFeatures(Keypoint& keypoint, vector<Keypoint>& features, const double *hist, int n, double mag_thr);
void DescriptorRepresentation(vector<Keypoint>& features, const vector<Mat>& gauss_pyr, int bins, int width);
double getFeatDist(Keypoint f1, Keypoint f2);
int findIndex(vector<double> a, double b);
bool checkEdge(Keypoint f, Mat &img);

bool FeatureCmp(Keypoint& f1, Keypoint& f2){
	return f1.scale < f2.scale;
}

void RtoG(const Mat &src, Mat &dst);
void Gaussian_Smooth(Mat &src, Mat &dst, double sigma);
void Normalize(Mat &src, Mat &dst);
void UpSampling(Mat &src, Mat &dst);
void DownSampling(Mat &src, Mat &dst);
void Difference(Mat &p, Mat &q, Mat &result);
int DeterminExtremePoints(int x, int y, vector<Mat>& dog_pyr, int index);
double PyrAt(vector<Mat>& pyr, int index, int x, int y);
void GetOffset(vector<Mat>& dog_pyr, int index, int x, int y, double offset_x[]);
void Hessian3(vector<Mat>& dog_pyr, int index, int x, int y, double *H);
bool InvertH3(double *H, double *H_inv);
bool passEdgeResponse(vector<Mat>& dog_pyr, int x, int y, int index);
void DogFirDiff(vector<Mat>& dog_pyr, int index, int x, int y, double *dx);
double GetFabsDx(vector<Mat>& dog_pyr, int index, int x, int y, double *offset_x);

void CalculateScale(vector<Keypoint>& keypoints);
void ToOrigin(vector<Keypoint>& keypoints);

void CreateInitGray(const Mat &src, Mat &dst);
void CreateGaussianPyramid(Mat &src, vector<Mat>& pyr, int Octaves);
void CreateDogPyramid(vector<Mat>& gauss_pyr, vector<Mat>& dog_pyr, int Octaves);
void LocalExtremePoints(vector<Mat>& dog_pyr, vector<Keypoint>& keypoints, int Octaves);
Keypoint* CorrectExtrmum(vector<Mat>& dog_pyr, int o, int s, int x, int y);



void PrintPixel(Mat &src);
void ShowPyramid(vector<Mat>& pyr);
void Printkeypoints(vector<Keypoint>& keypoints);
void Circlekeypoints(Mat &src, vector<Keypoint>& keypoints, char*);

void getSiftFeatures(Mat &src, vector<Keypoint>& feat);
int featureMatch(vector<Keypoint> & feat1, vector<Keypoint>& feat2, Mat &src, Mat &dst);

void Printfeatures(vector<Keypoint>& features);
void WriteFeatures(const vector<Keypoint>& features,const char* file);



void opencvMatch(Mat &img1, Mat &img2)
{
	cout << "============= OPENCV自带的SIFT算子匹配结果 ===========" <<endl;
	SiftFeatureDetector detector;
	SiftFeatureDetector extractor;
	BruteForceMatcher<L2<float>> matcher;
	clock_t start, end, t1;
	start = clock();
	vector<KeyPoint> keypoints1, keypoints2;
	detector.detect(img1, keypoints1);//检测img1中的SIFT特征点，存储到keypoints1中  
	detector.detect(img2, keypoints2);

	Mat descriptors1, descriptors2;
	extractor.compute(img1, keypoints1, descriptors1);
	t1 = clock();
	extractor.compute(img2, keypoints2, descriptors2);
	end = clock();

	cout << "图像1特征点个数:" << keypoints1.size() << endl;
	cout << "图像2特征点个数:" << keypoints2.size() << endl;
	cout << "SIFT算法用时（图1）： " << (t1 - start)*1.0 / CLOCKS_PER_SEC << "s"<<endl;
	cout << "SIFT算法用时（图2）： " << (end - t1)*1.0 / CLOCKS_PER_SEC << "s"<<endl;
	cout << "SIFT算法用时（2张图共计）： " << (end - start)*1.0 / CLOCKS_PER_SEC << "s"<<endl;

	//画出特征点  
	Mat img_keypoints1, img_keypoints2;
	drawKeypoints(img1, keypoints1, img_keypoints1, Scalar::all(-1), 0);
	drawKeypoints(img2, keypoints2, img_keypoints2, Scalar::all(-1), 0);
	imshow("OpenCV_Img1",img_keypoints1);  
	imshow("OpenCV_Img2",img_keypoints2);  

	//特征匹配  
	vector<vector<DMatch>> matches;//匹配结果  
	matcher.knnMatch(descriptors1, descriptors2, matches,2);//匹配两个图像的特征矩阵  
	vector<DMatch> goodmatches;
	int removed = 0;
	const float ratio = 0.6f;
	const float max_matching_distance = 250.f;
	vector<vector<DMatch>>::iterator matchi = matches.begin();
	for (; matchi != matches.end(); matchi++)
	{
		if (matchi->size() > 1)
		{
			if ((*matchi)[0].distance > ratio*(*matchi)[1].distance)
			{
				matchi->clear();
				removed++;
			}
			else if ((*matchi)[0].distance>max_matching_distance)
			{
				matchi->clear();
				removed++;
			}
			else
			{
				goodmatches.push_back((*matchi)[0]);
			}

		}
	}
	cout << "匹配点数：" << goodmatches.size() << endl;
	Mat img_matches;
	drawMatches(img1, keypoints1, img2, keypoints2, goodmatches, img_matches,
		Scalar::all(-1), CV_RGB(0, 255, 0), Mat(), 2);

	imshow("OpenCV_SIFT_Match", img_matches);
}

void Printfeatures(vector<Keypoint>& features){
	int size = features.size();
	cout << "The number of features is  " << size << endl;
	for (int i = 0; i < size; i++){
		cout << i; cout << "  ";
		cout << "octave:" << features[i].octave; cout << "  ";
		cout << "interval:" << features[i].interval; cout << "  ";
		cout << "dx:" << (int)features[i].dx; cout << "  ";
		cout << "dy:" << (int)features[i].dy; cout << "  ";
		cout << "scale:" << features[i].octave_scale << "  ori:" << features[i].ori << endl;
	}

/*
	vector<int> samepoints;
	for (int i = 0; i < size - 1; i++){
		for (int j = i + 1; j < size; j++){
			if ((features[i].dx == features[j].dx) && (features[i].dy == features[j].dy)){
				int p = i;
				samepoints.push_back(p);
			}
		}
	}

	for (int i = 0; i < samepoints.size(); i++){
		cout << samepoints[i] << "  " << endl;
	}
*/
}


void WriteFeatures(const vector<Keypoint>& features,const char* file)
{
	ofstream outfile;
	outfile.open(file);
	outfile << features.size() << endl;
	for (int i = 0; i < features.size(); i++)
	{
		outfile << features[i].scale << " " << features[i].dx << " " << features[i].dy << endl;
		for (int j = 0; j < Feature_Size; j++)
		{
			if (j % 20 == 0)
				outfile << endl;
			outfile << features[i].descriptor[j] << " ";
		}
		outfile << endl;
		outfile << endl;
	}
}
double round(double r)
{
	return (r>0.0) ? floor(r+0.5) : ceil(r-0.5);
}

bool CalcGradMagOri(Mat& gauss, int x, int y, double& mag, double& ori)
{
	if (x > 0 && x < gauss.cols - 1 && y > 0 && y < gauss.rows - 1)
	{
		double *data = (double*)gauss.data;
		int step = gauss.step / sizeof(*data);

		double dx = *(data + step*y + (x + 1)) - (*(data + step*y + (x - 1)));
		double dy = *(data + step*(y + 1) + x) - (*(data + step*(y - 1) + x));

		mag = sqrt(dx*dx + dy*dy);

		ori = atan2(dy, dx);
		return true;
	}
	else
		return false;
}

double* CalculateOrientationHistogram(Mat& gauss, int x, int y, int bins, int radius, double sigma)
{
	double *hist = new double[bins];

	for (int i = 0; i < bins; i++)
		*(hist + i) = 0.0;

	double mag, ori;
	double weight;

	int bin;
	const double PI2 = 2.0*CV_PI;

	double econs = -1.0 / (2.0*sigma*sigma);

	for (int i = -radius; i <= radius; i++)
	{
		for (int j = -radius; j <= radius; j++)
		{
			if (CalcGradMagOri(gauss, x + i, y + j, mag, ori))
			{
				weight = exp((i*i + j*j)*econs);

				bin = cvRound(bins * (CV_PI - ori) / PI2);
				bin = bin < bins ? bin : 0;

				hist[bin] += mag * weight;
			}
		}
	}
	return hist;
}


void OrientationAssignment(vector<Keypoint>& extrema, vector<Keypoint>& features, vector<Mat>& gauss_pyr){
	int n = extrema.size();
	double *hist;

	for (int i = 0; i < n; i++)
	{

		hist = CalculateOrientationHistogram(gauss_pyr[extrema[i].octave*(S + 3) + extrema[i].interval],
			extrema[i].x, extrema[i].y, ORI_HIST_BINS, cvRound(ORI_WINDOW_RADIUS*extrema[i].octave_scale),
			ORI_SIGMA_TIMES*extrema[i].octave_scale);

		for (int j = 0; j < ORI_SMOOTH_TIMES; j++)
			GaussSmoothOriHist(hist, ORI_HIST_BINS);
		double highest_peak = DominantDirection(hist, ORI_HIST_BINS);

		CalcOriFeatures(extrema[i], features, hist, ORI_HIST_BINS, highest_peak*ORI_PEAK_RATIO);

		delete[] hist;

	}
}

//高斯平滑，模板[0.25 0.5 0.25]
void GaussSmoothOriHist(double *hist, int n){
	double prev = hist[n - 1], temp, h0 = hist[0];

	for (int i = 0; i < n; i++)
	{
		temp = hist[i];
		hist[i] = 0.25 * prev + 0.5 * hist[i] +
			0.25 * (i + 1 >= n ? h0 : hist[i + 1]);
		prev = temp;
	}
}

//计算方向直方图的主方向
double DominantDirection(double *hist, int n){
	double maxd = hist[0];
	for (int i = 1; i < n; i++)
	{
		if (hist[i] > maxd)
			maxd = hist[i];
	}
	return maxd;
}

#define Parabola_Interpolate(l, c, r) (0.5*((l)-(r))/((l)-2.0*(c)+(r))) 

//抛物插值
void CalcOriFeatures(Keypoint& keypoint, vector<Keypoint>& features, const double *hist, int n, double mag_thr)
{
	double bin, PI2 = CV_PI * 2.0;
	int l, r;
	for (int i = 0; i < n; i++)
	{
		l = (i == 0) ? n - 1 : i - 1;
		r = (i + 1) % n;

		//hist[i]极值
		if (hist[i] > hist[l] && hist[i] > hist[r] && hist[i] >= mag_thr)
		{
			bin = i + Parabola_Interpolate(hist[l], hist[i], hist[r]);
			bin = (bin < 0) ? (bin + n) : (bin >= n ? (bin - n) : bin);
			Keypoint new_key;
			CopyKeypoint(keypoint, new_key);
			new_key.ori = ((PI2 * bin) / n) - CV_PI;
			features.push_back(new_key);
		}
	}
}

void CopyKeypoint(const Keypoint& src, Keypoint& dst)
{
	dst.dx = src.dx;
	dst.dy = src.dy;

	dst.interval = src.interval;
	dst.octave = src.octave;
	dst.octave_scale = src.octave_scale;
	dst.offset_interval = src.offset_interval;

	dst.offset_x = src.offset_x;
	dst.offset_y = src.offset_y;

	dst.ori = src.ori;
	dst.scale = src.scale;
	dst.mag = src.mag;
	dst.x = src.x;
	dst.y = src.y;
}





/**************************************************************************************************************/


//计算描述子
void DescriptorRepresentation(vector<Keypoint>& features, vector<Mat>& gauss_pyr, int bins, int width)
{
	double ***hist;

	for (int i = 0; i < features.size(); i++)
	{
		hist = CalculateDescrHist(gauss_pyr[features[i].octave*(S + 3) + features[i].interval],
			features[i].x, features[i].y, features[i].octave_scale, features[i].ori, bins, width);


		HistToDescriptor(hist, width, bins, features[i]);

		for (int j = 0; j < width; j++){
			for (int k = 0; k < width; k++){
				delete[] hist[j][k];
			}
			delete[] hist[j];
		}
		delete[] hist;
	}
}


double*** CalculateDescrHist(Mat& gauss, int x, int y, double octave_scale, double ori, int bins, int width)
{
	double ***hist = new double**[width];

	//初始化
	for (int i = 0; i < width; i++)
	{
		hist[i] = new double*[width];
		for (int j = 0; j < width; j++)
		{
			hist[i][j] = new double[bins];
		}
	}

	for (int r = 0; r < width; r++)
		for (int c = 0; c < width; c++)
			for (int o = 0; o < bins; o++)
				hist[r][c][o] = 0.0;


	double cos_ori = cos(ori);
	double sin_ori = sin(ori);

	double sigma = 0.5 * width;
	double conste = -1.0 / (2 * sigma*sigma);

	double PI2 = CV_PI * 2;

	double sub_hist_width = 3 * octave_scale;

	//半径
	int radius = (sub_hist_width*sqrt(2.0)*(width + 1)) / 2.0 + 0.5; //四舍五入

	double grad_ori, grad_mag;
	for (int i = -radius; i <= radius; i++)
	{
		for (int j = -radius; j <= radius; j++)
		{
			double rot_x = (cos_ori * j - sin_ori * i) / sub_hist_width;
			double rot_y = (sin_ori * j + cos_ori * i) / sub_hist_width;


			//判断落在4*4窗口
			double xbin = rot_x + width / 2 - 0.5;
			double ybin = rot_y + width / 2 - 0.5;

			//
			if (xbin > -1.0 && xbin < width && ybin > -1.0 && ybin < width){
				if (CalcGradMagOri(gauss, x + j, y + i, grad_mag, grad_ori)){
					grad_ori = (CV_PI - grad_ori) - ori;
					while (grad_ori < 0.0)
						grad_ori += PI2;
					while (grad_ori >= PI2)
						grad_ori -= PI2;

					double obin = grad_ori * (bins / PI2);

					double weight = exp(conste*(rot_x*rot_x + rot_y * rot_y));

					InterpHistEntry(hist, xbin, ybin, obin, grad_mag*weight, bins, width);

				}
			}
		}
	}

	return hist;
}

void InterpHistEntry(double ***hist, double xbin, double ybin, double obin, double mag, int bins, int d){
	double d_r, d_c, d_o, v_r, v_c, v_o;
	double** row, *h;
	int r0, c0, o0, rb, cb, ob, r, c, o;

	r0 = cvFloor(ybin);
	c0 = cvFloor(xbin);
	o0 = cvFloor(obin);
	d_r = ybin - r0;
	d_c = xbin - c0;
	d_o = obin - o0;

	for (r = 0; r <= 1; r++)
	{
		rb = r0 + r;
		if (rb >= 0 && rb < d)
		{
			v_r = mag * ((r == 0) ? 1.0 - d_r : d_r);
			row = hist[rb];
			for (c = 0; c <= 1; c++)
			{
				cb = c0 + c;
				if (cb >= 0 && cb < d)
				{
					v_c = v_r * ((c == 0) ? 1.0 - d_c : d_c);
					h = row[cb];
					for (o = 0; o <= 1; o++)
					{
						ob = (o0 + o) % bins;
						v_o = v_c * ((o == 0) ? 1.0 - d_o : d_o);
						h[ob] += v_o;
					}
				}
			}
		}
	}


}

void HistToDescriptor(double ***hist, int width, int bins, Keypoint& feature){
	int int_val, i, r, c, o, k = 0;

	for (r = 0; r < width; r++)
		for (c = 0; c < width; c++)
			for (o = 0; o < bins; o++)
			{
				feature.descriptor[k++] = hist[r][c][o];
			}

			feature.descr_length = k;
			NormalizeDescr(feature);
			for (i = 0; i < k; i++)
				if (feature.descriptor[i] > DESCR_MAG_THR)
					feature.descriptor[i] = DESCR_MAG_THR;
			NormalizeDescr(feature);

			/* convert floating-point descriptor to integer valued descriptor */

			for (i = 0; i < k; i++)
			{
				int_val = INT_DESCR_FCTR * feature.descriptor[i];
				feature.descriptor[i] = min(255, int_val);
			}

}

void NormalizeDescr(Keypoint& feat){
	double cur, len_inv, len_sq = 0.0;
	int i, d = feat.descr_length;

	for (i = 0; i < d; i++){
		cur = feat.descriptor[i];
		len_sq += cur*cur;
	}
	len_inv = 1.0 / sqrt(len_sq);
	for (i = 0; i < d; i++)
		feat.descriptor[i] *= len_inv;
}

bool checkEdge(Keypoint f, Mat &img)
{
	Size img_size = img.size();
	int width = img_size.width, height = img_size.height;
	int dx = f.dx, dy = f.dy;
	int delta = 3;
	if ((dx<delta && dy<delta) || (dx>(width-delta) && dy>(height-delta)) || (dx<delta && (dy>height-delta)) || (dx>(width-delta) && dy<delta))
		return true;
	else
	{
		return false;
	}
}

int featureMatch(vector<Keypoint> & feat1, vector<Keypoint>& feat2, Mat &src, Mat &dst)
{
	int numFeat1 = feat1.size();
	int numFeat2 = feat2.size();
	Size src_size = src.size();
	Size dst_size = dst.size();

	int dwdith = src_size.width/2;
	//int dheight = 0;
	int dheight = src_size.height*0.66;
	Size img_size = Size(src_size.width+dst_size.width, dheight+dst_size.height);

	Mat img_new(img_size.height, img_size.width, src.type() );
	Mat part;
	part = img_new(cv::Rect(0,0,src_size.width,src_size.height)); 
	src.copyTo(part);
	part = img_new(cv::Rect(src_size.width,dheight,dst_size.width,dst_size.height)); 
	dst.copyTo(part);
	int totalMatch = 0;

	for (int i=0; i<numFeat1; i++)
	{
		Keypoint f1 = feat1[i];
		if (checkEdge(f1, src))
			continue;
		vector<double> Dist;
		for (int j=0; j<numFeat2; j++)
		{
			Keypoint f2 = feat2[j];
			Dist.push_back(getFeatDist(f1, f2));
		}
		vector<double> D1(Dist);
		sort(Dist.begin(), Dist.end());
		double Min1 = Dist.at(0);
		double Min2 = Dist.at(1);
		if ((Min2-Min1)/Min2 > MATCH_THR)
		{
			int idx = findIndex(D1, Min1);
			CvPoint center1 = cvPoint(f1.dx, f1.dy);
			CvPoint center2 = cvPoint(feat2[idx].dx+src_size.width, feat2[idx].dy+dheight);
			CvScalar color = CV_RGB(255, 0, 0);
			circle(img_new, center1, 3, color, 1, 8, 0);
			circle(img_new, center2, 3, color, 1, 8, 0);
			line(img_new, center1, center2, Scalar(255,0,0), 2);
			totalMatch ++;
		}
		
	}
	imshow("MySIFT_Match", img_new);
	return totalMatch;
}

int findIndex(vector<double> a, double b)
{
	for (int i=0; i<a.size(); i++)
	{
		if ((a.at(i)-b)<0.0000001)
			return i;
	}
}

double getFeatDist(Keypoint f1, Keypoint f2)
{
	double *data1 = f1.descriptor;
	double *data2 = f2.descriptor;
	double dist = 0;
	for (int i=0; i<Feature_Size; i++)
	{
		dist += (data1[i]-data2[i])*(data1[i]-data2[i]);
	}
	dist = sqrt(dist);
	return dist;
}


void getSiftFeatures(Mat &src, vector<Keypoint>& feat)
{
	Mat dst, init;
	CreateInitGray(src, init);

	vector<Mat> gauss_pyr, dog_pyr;
	int Octaves = log((double)min(init.rows, init.cols)) / log(2.0) - TopLevel;

	CreateGaussianPyramid(init, gauss_pyr, Octaves);
	CreateDogPyramid(gauss_pyr, dog_pyr, Octaves);

	vector<Keypoint> keypoints;
	LocalExtremePoints(dog_pyr, keypoints, Octaves);
	CalculateScale(keypoints);
	ToOrigin(keypoints);

	OrientationAssignment(keypoints, feat, gauss_pyr);
	DescriptorRepresentation(feat, gauss_pyr, 8, 4);
	sort(feat.begin(), feat.end(), FeatureCmp);
}

void CreateInitGray(const Mat &src, Mat &dst){
	Mat gra, gra_float, upmat;
	RtoG(src, gra);
	Normalize(gra, gra_float);
	UpSampling(gra_float, upmat);
	double sigma_1 = sqrt(sigma0*sigma0 - sigma_pic0*sigma_pic0);
	Gaussian_Smooth(upmat, dst, sigma_1);
}



void Gaussian_Smooth(Mat &src, Mat &dst, double sigma){
	dst.create(src.size(), src.type());
	int GKsize;//Gaussian kernel size
	int center_len;//center length
	double sum = 0;//normalization

	if (sigma <= 0||src.channels()!=1){
		cout << "GaussianSmooth error!!" << endl;
		return;
	}

	GKsize = round(sigma * 3) * 2 + 1;
	center_len = (GKsize - 1) / 2;
	double *GK_Mat = new double[GKsize*GKsize];//mat of kernel
	double A = 1 / (2 * PI*sigma*sigma), B = -1 / (2 * sigma*sigma);

	for (int i = 0; i < GKsize; i++){
		for (int j = 0; j < GKsize; j++){
			int x = i - (GKsize - 1) / 2;
			int y = j - (GKsize - 1) / 2;
			GK_Mat[i*GKsize + j] = A*exp(B*(x*x + y*y));
			sum += GK_Mat[i*GKsize + j];
		}
	}

	for (int i = 0; i < GKsize*GKsize; i++)
		GK_Mat[i] /= sum;

	double* Data_dst = (double*)dst.data;
	double* Data_src = (double*)src.data;
	int Step_src = src.step / sizeof(Data_src[0]);
	int Step_dst = dst.step / sizeof(Data_dst[0]);

	//rows equals to size().y,colsequals to size().x
	for (int y = 0; y < src.rows; y++){
		for (int x = 0; x < src.cols; x++){
			double gray = 0;
			for (int j = -center_len; j <= center_len; j++){
				for (int i = -center_len; i <= center_len; i++){
					if (i + x < 0 || i + x >= src.cols) continue;
					if (j + y < 0 || j + y >= src.rows) continue;
						gray += *(Data_src + (y + j)*Step_src + (x + i))*GK_Mat[(j + center_len)*GKsize + i + center_len];
				}
			}
				*(Data_dst + y*Step_dst + x) = gray;
		}
	}
	delete[]GK_Mat;
}


void RtoG(const Mat &src, Mat &dst){
	if (dst.empty())
		dst.create(src.size(), CV_8U);
	uchar* Data_src = src.data;
	uchar* Data_dst = dst.data;

	for (int y = 0; y < dst.cols; y++){
		for (int x = 0; x < dst.rows; x++){
			double b = *(Data_src + x*src.step + y*src.channels() + 0);
			double g = *(Data_src + x*src.step + y*src.channels() + 1);
			double r = *(Data_src + x*src.step + y*src.channels() + 2);
			*(Data_dst + x*dst.step + y*dst.channels()) = cvRound((r * 0.3 + g * 0.59 + b * 0.11));
		}
	}
}





/*********************************************************************************************************************/
//PRINT!!!
void PrintPixel(Mat &src){
	double a;
	double* data = (double*)src.data;
	int Step = src.step / sizeof(data[0]);
	for (int y = 0; y < src.rows; y++){
		for (int x = 0; x < src.cols; x++){
			*(data + y*Step + x) = abs(*(data + y*Step + x))+0.3;
			//printf("%lf  ", a);
		}
	}
}

void ShowPyramid(vector<Mat>& pyr){
	imshow("image0", pyr[0 * 5 + 0]);
	imshow("image1", pyr[1 * 5 + 0]);
	imshow("image2", pyr[2 * 5 + 0]);
	imshow("image3", pyr[3 * 5 + 0]);
	imshow("image4", pyr[4 * 5 + 0]);
	imshow("image5", pyr[5 * 5 + 0]);
	imshow("image6", pyr[6 * 5 + 0]);
	//imshow("image7", pyr[7 * 5 + 0]);
	
	//imwrite("image0.jpg", pyr[0 * 6 + 5]);
	//imwrite("image1.jpg", pyr[1 * 6 + 5]);
	//imwrite("image2.jpg", pyr[2 * 6 + 5]);
	//imwrite("image3.jpg", pyr[3 * 6 + 5]);
	//imwrite("image4.jpg", pyr[4 * 6 + 5]);
	//imwrite("image5.jpg", pyr[5 * 6 + 5]);
	
}


void Printkeypoints(vector<Keypoint>& keypoints){
	int size = keypoints.size();
	cout << "The number of keypoints is  " << size << endl;
	for (int i = 0; i < size; i++){
		printf("keypoints %d  ", i);
		cout << "x:" << keypoints[i].x << "  ";
		cout << "y:" << keypoints[i].y << "  ";
		cout << "octave:" << keypoints[i].octave << "  ";
		cout << "interva:" << keypoints[i].interval << "  ";
		cout << "dx:" << keypoints[i].dx << "  ";
		cout << "dy:" << keypoints[i].dy << "  " << endl;
	}
}


void Circlekeypoints(Mat &src, vector<Keypoint>& keypoints, char* WindowName){
	int pointnum = keypoints.size();
	for (int i = 0; i < pointnum; i++){
		CvPoint center = cvPoint(keypoints[i].dx, keypoints[i].dy);
		CvScalar color = CV_RGB(255, 0, 0);
		circle(src, center, 3, color, 1, 8, 0);
		i++;
	}
	imshow(WindowName, src);
}



/**********************************************************************************************************************/



void Normalize(Mat &src, Mat &dst){
	dst.create(src.size(), CV_64FC1);
	double* Data_dst = (double*)dst.data;
	int Step_dst = dst.step / sizeof(Data_dst[0]);
	for (int y = 0; y < dst.rows; y++){
		for (int x = 0; x < dst.cols; x++){
			*(Data_dst + y*Step_dst + x) = *(src.data + y*src.step + x) / 255.0;
			//cout << *(src.data + y*src.step + x) << endl;
		}
	}
}


void UpSampling(Mat &src, Mat &dst){
	if (src.channels() != 1){
		cout << "UpSampling error!!" << endl;
		return;
	}
	dst.create(src.rows * 2, src.cols * 2, src.type());
	double* Data_src = (double*)src.data;
	double* Data_dst = (double*)dst.data;
	int Step_src = src.step / sizeof(Data_src[0]);
	int Step_dst = dst.step / sizeof(Data_dst[0]);
	int i = 0, j = 0;
	for (int y = 0, j = 0; y < src.rows - 1; y++, j += 2){
		for (int x = 0,i = 0; x < src.cols - 1; x++, i += 2){
			double same = *(Data_src + y*Step_src + x);
			double r = (*(Data_src + y*Step_src + x) + *(Data_src + y*Step_src + x + 1))/2.0;
			double d = (*(Data_src + (y + 1)*Step_src + x) + *(Data_src + (y + 1)*Step_src + x)) / 2.0;
			double rd = (*(Data_src + y*Step_src + x) + *(Data_src + y*Step_src + x + 1) + *(Data_src + 
				        (y + 1)*Step_src + x) + *(Data_src + (y + 1)*Step_src + x+1)) / 4.0;

			*(Data_dst + j*Step_dst + i) = same;
			*(Data_dst + j*Step_dst + i + 1) = r;
			*(Data_dst + (j + 1)*Step_dst + i) = d;
			*(Data_dst + (j + 1)*Step_dst + i + 1) = rd;
		}
	}
	for (int y = 0; y < dst.rows; y++){
		*(Data_dst + y*Step_dst + dst.cols - 2) = *(Data_dst + y*Step_dst + dst.cols - 3);
		*(Data_dst + y*Step_dst + dst.cols - 1) = *(Data_dst + y*Step_dst + dst.cols - 3);
	}
	for (int x = 0; x < dst.cols; x++){
		*(Data_dst + (dst.rows - 2)*Step_dst + x) = *(Data_dst + (dst.rows - 3)*Step_dst + x);
		*(Data_dst + (dst.rows - 1)*Step_dst + x) = *(Data_dst + (dst.rows - 3)*Step_dst + x);
	}
}


void DownSampling(Mat &src, Mat &dst){
	if (src.channels() != 1)
		return;
	if (src.rows == 1 || src.cols == 1){
		src.copyTo(dst);
		return;
	}
	dst.create((int)(src.rows / 2), (int)(src.cols / 2), src.type());

	double* Data_src = (double*)src.data;
	double* Data_dst = (double*)dst.data;
	int Step_src = src.step / sizeof(Data_src[0]);
	int Step_dst = dst.step / sizeof(Data_dst[0]);

	for (int y = 0; y < dst.rows; y++){
		for (int x = 0; x < dst.cols; x++){
			if ((2 * x < src.cols) && (2 * y < src.rows)){
				*(Data_dst + y*Step_dst + x) = *(Data_src + 2*y*Step_src + 2*x);
			}
		}
	}


}


void Difference(Mat &p, Mat &q, Mat &result){
	if ((p.rows != q.rows) || (p.cols != q.cols) || q.type() != q.type())
		return;
	if (!result.empty())
		return;
	result.create(p.size(), p.type());
	
	double* Data_p = (double*)p.data;
	double* Data_q = (double*)q.data;
	double* Data_result = (double*)result.data;
	int Step_p = p.step / sizeof(Data_p[0]);
	int Step_q = q.step / sizeof(Data_q[0]);
	int Step_result = result.step / sizeof(Data_result[0]);

	for (int y = 0; y < p.rows; y++){
		for (int x = 0; x < p.cols; x++){
			*(Data_result + y*Step_result + x) = *(Data_p + y*Step_p + x) - *(Data_q + y*Step_q + x);
		}
	}

}

void CreateDogPyramid(vector<Mat>& gauss_pyr, vector<Mat>& dog_pyr, int Octaves){
	for (int o = 0; o < Octaves; o++){
		for (int s = 1; s < S + 3; s++){
			Mat temp;
			Difference(gauss_pyr[o*(S + 3) + s], gauss_pyr[o*(S + 3) + s - 1], temp);
			dog_pyr.push_back(temp);
		}
	}


}




void CreateGaussianPyramid(Mat &src, vector<Mat>& gauss_pyr, int Octaves){
	double k = pow(2, 1.0 / S);
	double *sigma_s = new double[S+3];
	sigma_s[0] = sigma0;
	for (int i = 1; i < S + 3; i++){
		sigma_s[i] = sigma0*sqrt(pow(k, i)*pow(k, i) - pow(k, i - 1)*pow(k, i - 1));
	}
	for (int o = 0; o < Octaves; o++){
		for (int s = 0; s < S + 3; s++){
			Mat temp;
			if ((o == 0) && (s == 0)){
				src.copyTo(temp);
			}
			else if ((o != 0) && (s == 0)){
				DownSampling(gauss_pyr[o*(S + 3) - 2], temp);
			}
			else{
				Gaussian_Smooth(gauss_pyr[o*(S + 3) + s - 1], temp, sigma_s[s]);
			}
			gauss_pyr.push_back(temp);
		}
	}

}


int DeterminExtremePoints(int x, int y, vector<Mat>& dog_pyr, int index){
	double* data = (double*)dog_pyr[index].data;
	int Step = dog_pyr[index].step / sizeof(data[0]);
	double Mag = *(data + y*Step + x);
	int count_max = 0, count_min = 0;
	
/*******************************************************************************************************************
	if (Mag > 0)
	{
		for (int i = -1; i <= 1; i++)
		{
			int stp = dog_pyr[index + i].step / sizeof(double);
			for (int j = -1; j <= 1; j++)
			{
				for (int k = -1; k <= 1; k++)
				{
					//检查极大值
					if (Mag < *((double*)dog_pyr[index + i].data + stp*(y + j) + (x + k)))
					{
						return 0;
					}
				}
			}
		}
	}
	else
	{
		for (int i = -1; i <= 1; i++)
		{
			int stp = dog_pyr[index + i].step / sizeof(double);
			for (int j = -1; j <= 1; j++)
			{
				for (int k = -1; k <= 1; k++)
				{
					//检查极小值
					if (Mag > *((double*)dog_pyr[index + i].data + stp*(y + j) + (x + k)))
					{
						return 0;
					}
				}
			}
		}
	}

	return 1;
************************************************************************************************************************/

	for (int k = -1; k <= 1; k++){
		double* data_o = (double*)dog_pyr[index+k].data;
		int step_o = dog_pyr[index].step / sizeof(data_o[0]);
		for (int j = -1; j <= 1; j++){
			for (int i = -1; i <= 1; i++){
				if (Mag > *(data_o + (y + j)*step_o + x + i))
					count_max += 1;
				if (Mag < *(data_o + (y + j)*step_o + x + i))
					count_min += 1;
			}
		}
	}

	if (count_max == 26 || count_min == 26)
		return 1;
	else
		return 0;
}    


void LocalExtremePoints(vector<Mat>& dog_pyr, vector<Keypoint>& keypoints, int Octaves){
	for (int o = 0; o < Octaves; o++){
		for (int s = 1; s < S + 1; s++){
			int index = o*(S + 2) + s;
			double* data = (double*)dog_pyr[index].data;
			int step = dog_pyr[index].step / sizeof(data[0]);
			
			for (int y = 1; y < dog_pyr[index].rows - 1; y++){
				for (int x = 1; x < dog_pyr[index].cols - 1; x++){
					if (fabs(*(data + y*step + x))>0.5*Thresh / S){
						if (DeterminExtremePoints(x, y, dog_pyr, index) == 1){
							Keypoint* extrmum = CorrectExtrmum(dog_pyr, o, s, x, y);

							if (extrmum){
								if (passEdgeResponse(dog_pyr, extrmum->x, extrmum->y, index)){
									extrmum->mag = *(data + extrmum->y*step + extrmum->x);
									keypoints.push_back(*extrmum);
								}
							}
							delete extrmum;
						}
					}
				}
			}

		}
	}
}


Keypoint* CorrectExtrmum(vector<Mat>& dog_pyr, int o, int s, int x, int y){
	int index = o*(S + 2) + s;
	int idx = o*(S + 2) + s;
	double offset_x[3] = { 0 };
	int k = 0;
	int intvl = s;
	Mat &mat = dog_pyr[index];

	while (k < 5){
		GetOffset(dog_pyr, idx, x, y, offset_x);
		if (fabs(offset_x[0]) < 0.5&&fabs(offset_x[1]) < 0.5&&fabs(offset_x[2]) < 0.5)
			break;

		x += cvRound(offset_x[0]);
		y += cvRound(offset_x[1]);
		s += cvRound(offset_x[2]);

		idx = index - intvl + s;

		if (s<1 || s>S || x < 1 || x >= mat.cols - 1 || y < 1 || y >= mat.rows)
			return NULL;
		k++;
	}

	if (k >= 5)
		return NULL;

	if (GetFabsDx(dog_pyr, idx, x, y, offset_x) < Thresh / S)
		return NULL;

	Keypoint *keypoint = new Keypoint;

	keypoint->x = x;
	keypoint->y = y;
	keypoint->interval = s;
	keypoint->octave = o;

	keypoint->offset_x = offset_x[0];
	keypoint->offset_y = offset_x[1];
	keypoint->offset_interval = offset_x[2];

	keypoint->dx = (x + offset_x[0])*pow(2.0, o );
	keypoint->dy = (y + offset_x[1])*pow(2.0, o );

	return keypoint;
}


double PyrAt(vector<Mat>& pyr, int index, int x, int y)
{
	double *data = (double*)pyr[index].data;
	int step = pyr[index].step / sizeof(data[0]);
	double val = *(data + y*step + x);

	return val;
}

#define At(index, x, y) (PyrAt(dog_pyr, (index), (x), (y)))

void GetOffset(vector<Mat>& dog_pyr, int index, int x, int y, double offset_x[]){
	double H[9] = { 0.0 };
	double H_inv[9] = { 0.0 };
	Hessian3(dog_pyr, index, x, y, H);
	InvertH3(H, H_inv);
	double dx[3] = { 0.0 };//D的一阶偏导
	DogFirDiff(dog_pyr, index, x, y, dx);

	for (int i = 0; i < 3; i++){
		offset_x[i] = 0.0;
		for (int j = 0; j < 3; j++){
			offset_x[i] += H_inv[i * 3 + j] * dx[j];
		}
		offset_x[i] = -offset_x[i];
	}



}


#define Hat(i, j) (*(H+(i)*3 + (j)))
#define HIat(i, j) (*(H_inv+(i)*3 + (j)))

void Hessian3(vector<Mat>& dog_pyr, int index, int x, int y, double *H){
	double Dxx, Dyy, Dss, Dxy, Dys, Dsx;
	double val = At(index, x, y);

	Dxx = At(index, x + 1, y) + At(index, x - 1, y) - 2 * val;
	Dyy = At(index, x, y + 1) + At(index, x, y - 1) - 2 * val;
	Dss = At(index + 1, x, y) + At(index - 1, x, y) - 2 * val;

	Dxy = (At(index, x + 1, y + 1) + At(index, x - 1, y - 1)
		- At(index, x + 1, y - 1) - At(index, x - 1, y + 1)) / 4.0;

	Dsx = (At(index + 1, x + 1, y) + At(index - 1, x - 1, y)
		- At(index - 1, x + 1, y) - At(index + 1, x - 1, y)) / 4.0;

	Dys = (At(index + 1, x, y + 1) + At(index - 1, x, y - 1)
		- At(index + 1, x, y - 1) - At(index - 1, x, y + 1)) / 4.0;
	
	Hat(0, 0) = Dxx;
	Hat(1, 1) = Dyy;
	Hat(2, 2) = Dss;
	Hat(0, 1) = Dxy; Hat(1, 0) = Dxy;
	Hat(0, 2) = Dsx; Hat(2, 0) = Dsx;
	Hat(1, 2) = Dys; Hat(2, 1) = Dys;
}

bool InvertH3(double *H, double *H_inv){
	//求H得行列式值，若行列式为0，逆不存在
	double det_H = Hat(0, 0)*Hat(1, 1)*Hat(2, 2)+ Hat(0, 1)*Hat(1, 2)*Hat(2, 0)
				 + Hat(0, 2)*Hat(1, 0)*Hat(2, 1)- Hat(0, 0)*Hat(1, 2)*Hat(2, 1)
				 - Hat(0, 1)*Hat(1, 0)*Hat(2, 2)- Hat(0, 2)*Hat(1, 1)*Hat(2, 0);
	if (fabs(det_H) < 1e-10)
		return false;

	//逆存在，求H得逆
	HIat(0, 0) = Hat(1, 1) * Hat(2, 2) - Hat(2, 1)*Hat(1, 2);
	HIat(0, 1) = -(Hat(0, 1) * Hat(2, 2) - Hat(2, 1) * Hat(0, 2));
	HIat(0, 2) = Hat(0, 1) * Hat(1, 2) - Hat(0, 2)*Hat(1, 1);

	HIat(1, 0) = Hat(1, 2) * Hat(2, 0) - Hat(2, 2)*Hat(1, 0);
	HIat(1, 1) = -(Hat(0, 2) * Hat(2, 0) - Hat(0, 0) * Hat(2, 2));
	HIat(1, 2) = Hat(0, 2) * Hat(1, 0) - Hat(0, 0)*Hat(1, 2);

	HIat(2, 0) = Hat(1, 0) * Hat(2, 1) - Hat(1, 1)*Hat(2, 0);
	HIat(2, 1) = -(Hat(0, 0) * Hat(2, 1) - Hat(0, 1) * Hat(2, 0));
	HIat(2, 2) = Hat(0, 0) * Hat(1, 1) - Hat(0, 1)*Hat(1, 0);

	for (int i = 0; i < 9; i++){
		*(H_inv + i) /= det_H;
	}
		return true;

}

void DogFirDiff(vector<Mat>& dog_pyr, int index, int x, int y, double *dx){
	double Dx, Dy, Ds;
	
	Dx = (At(index, x + 1, y) - At(index, x - 1, y)) / 2.0;
	Dy = (At(index, x, y + 1) - At(index, x, y - 1)) / 2.0;
	Ds = (At(index + 1, x, y) - At(index - 1, x, y)) / 2.0;

	dx[0] = Dx;
	dx[1] = Dy;
	dx[2] = Ds;

}

double GetFabsDx(vector<Mat>& dog_pyr, int index, int x, int y, double *offset_x){
	//|D(x^)|=D + 0.5 * dx * offset_x; dx=(Dx, Dy, Ds)^T
	double dx[3];
	DogFirDiff(dog_pyr, index, x, y, dx);

	double term = 0.0;
	for (int i = 0; i < 3; i++)
		term += dx[i] * offset_x[i];

	double *data = (double *)dog_pyr[index].data;
	int step = dog_pyr[index].step / sizeof(data[0]);
	double val = *(data + y*step + x);

	return fabs(val + 0.5 * term);
}


bool passEdgeResponse(vector<Mat>& dog_pyr, int x, int y, int index){
	double *data = (double *)dog_pyr[index].data;
	int Step = dog_pyr[index].step / sizeof(data[0]);
	double Mag = *(data + y*Step + x);
	double Dxx, Dyy, Dxy;
	double Tr_H, Det_H;
	Dxx = *(data + y*Step + (x + 1)) + *(data + y*Step + (x - 1)) - *(data + y*Step + x) * 2;
	Dyy = *(data + (y + 1)*Step + x) + *(data + (y - 1)*Step + x) - *(data + y*Step + x) * 2;
	Dxy = (*(data + (y + 1)*Step + (x + 1)) + *(data + (y - 1)*Step + (x - 1)) - *(data + (y + 1)*Step + (x - 1)) - *(data + (y - 1)*Step + (x + 1))) / 4.0;
	Tr_H = Dxx + Dyy;
	Det_H = Dxx*Dyy - Dxy*Dxy;

	if (Tr_H < 0)
		return false;

	if (Tr_H*Tr_H / Det_H < (R + 1)*(R + 1) / R)
		return true;

	return false;
}


void CalculateScale(vector<Keypoint>& keypoints){
	double intvl = 0;
	for (int i = 0; i < keypoints.size(); i++)
	{
		intvl = keypoints[i].interval + keypoints[i].offset_interval;
		keypoints[i].scale = sigma0 * pow(2.0, keypoints[i].octave + intvl / S);
		keypoints[i].octave_scale = sigma0 * pow(2.0, intvl / S);
	}

}

void ToOrigin(vector<Keypoint>& keypoints){
	for (int i = 0; i < keypoints.size(); i++)
	{
		keypoints[i].dx /= 2;
		keypoints[i].dy /= 2;
		keypoints[i].scale /= 2;
	}

}
#endif
