#ifndef VALIDATION_H
#define VALIDATION_H


#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <cstring>
#include <iostream>
#include <fstream>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

class analysis
{
public:
    analysis(int, double);
    void parserLabels();
    bool measure(cv::Rect rect_pred, int n_frm, int& _op, int& _dp, ofstream& frms);
    int cuentaTruthBoxes(int total_n_frms);
    void closeFile_frms();

private:
    double threshold_op;
    //ofstream frms;
    int threshold_dp;
    int prev_nfrm;
    double acum_sqr;
    double rms_prev_frm;
    short int prev_num;
    bool overlapParcial( cv::Rect r1, cv::Rect r2, double min_overlap );
    bool verifyMassCenters(Point2f& p, Point2f& q);
    Point2f calculateMassCenter( Mat roi );
    double euclideanDist(Point2f p, Point2f q);
    double calculateIoU( Rect r1, Rect r2);
    Point2f calculateRectCenter( cv::Rect rec );
};


#endif // VALIDATION_H
