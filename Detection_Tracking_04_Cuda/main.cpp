#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <stdlib.h>
#include <parallel/numeric>

#include <dlib/image_processing.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/dir_nav.h>

#include "compareD.h"
#include "validation.h"


#include "opencv2/core/core.hpp"
#include "/usr/local/include/opencv2/core/hal/hal.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/optflow/motempl.hpp"


#include "opencv2/cudafilters.hpp"
#include "opencv2/cudaobjdetect.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudawarping.hpp"

#include <numeric>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h> //mkdir

# include <omp.h>


using namespace std;
using namespace cv;
using namespace cv::cuda;
using namespace dlib;

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

string name_vid = "20160823065240_20160823065320.avi";
//string name_vid = "20160823065348_20160823065402.avi";
//string name_vid = "video3.avi";

struct MyList {
   Rect box;
   float angle;
   Point ctr_masa;
   short int porc_acep;
   short int porc_vida;
   dlib::correlation_tracker tracker;
   bool tracking;
};
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void function_detect_motion( std::vector<Rect>& mov_objs, std::vector<double>& seg_angles, Mat& frame, Mat& prev_frame, Mat& motion_history);
void function_detect_object( std::vector<Rect>& det_objs, Mat& frame_vis);
void function_matching_motion_object(std::vector<double>& det_angles, std::vector<double>& seg_angles, std::vector<Rect>& det_objs, std::vector<Rect>& mov_objs );
void function_tracking(Mat& frame, Mat& frame_vis,
                       std::vector<double>& det_angles,
                       std::vector<double>& seg_angles,
                       std::vector<Rect>& det_objs,
                       std::vector<MyList>& mem_objs,
                       ofstream& fRms, ofstream& fIoU, ofstream& ftiempo,
                       bool& analy,
                       int det_objs_sz, int mem_objs_sz,
                       ofstream& n_predbox,
                       analysis& analysis_,
                       int acum_op, int acum_dp );

static cv::Rect dlibRectangleToOpenCV(dlib::rectangle r);
static dlib::rectangle openCVRectToDlib(cv::Rect r);
std::vector<Rect> detect1( Mat img );
std::vector<Rect> detect1_gpu(Mat img);
bool overlapTotal( Rect, Rect );
bool overlapParcial( Rect r1, Rect r2, double );
std::vector<Rect> movDetect( Mat, Mat, int, std::vector<double>&, Mat& );
void dibujaObjs( std::vector<Rect>, Mat&, Scalar color );
void dibujaObj( Rect, Mat&, Scalar color );
void printMat(Mat m, int w, int h );
void writeMatToFile(cv::Mat& m, const char* filename);
Point calculateMassCenter( Mat img );
bool enelAmbitoGlobal(Point punto , int width, int height);
bool isplate( Rect&, Rect& );
bool iscap( Rect&, Rect& );
void convertAndResize(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& gray, cv::cuda::GpuMat& resized, double scale);
void saveThreshold(Mat frame, Rect this_rect, int c_th, int c);
int carColorExtraction8binsHSV(Mat &img);
void calcMotionGradient_real( InputArray _mhi, OutputArray _mask, OutputArray _orientation, double delta1, double delta2, int aperture_size );
void calcMotionGradient_gpu( InputArray _mhi, OutputArray _mask, OutputArray _orientation,double delta1, double delta2, int aperture_size );

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
string cascadeName = "../Cascades/cascade.xml";
cv::CascadeClassifier cascada;
Ptr<cv::cuda::CascadeClassifier> cascade_gpu = cv::cuda::CascadeClassifier::create(cascadeName);
bool t1 = cascada.load(cascadeName);
bool USE_GPU = false;
VideoWriter outputVideo;

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
int red_frm = 2;
int w =  720, h = 480, c = 0; //int w =  1920, h = 1080, c = 0;
int w_mtn = w/2, h_mtn = h/2;
int izqx = 1; // el 20% del lado izquierdo de la escena debe ser excluida en la detección.
int derx = 99; // Más allá del 78% del ancho de la imagen se excluye de la detección.
int supy_track = 2; // Menos del 1% de la altura de la imagen, se excluye para el tracking.
int infy_track = 99; // Más del 88% de la altura de la imagen, se excluye para el tracking.
int x_lim_izq = (w * izqx) / 100;
int x_lim_der = (w * derx) / 100;
int y_lim_sup_track = (h * supy_track) / 100;
int y_lim_inf_track = (h * infy_track) / 100;

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
float MHI_DURATION = 0.5;
int DEFAULT_THRESHOLD = 5; // mientras sea menor más tolerante a bordes será
float MAX_TIME_DELTA = 1000000.0; // dont work 0.25; mientras más grande el valor este se vuelve más aceptable a movimientos lentos
float MIN_TIME_DELTA = 5.0; // dont work 0.05;
int APERTURE_SZ_SOBEL= 3;
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


//std::vector< std::vector<float>> mem_objs;
bool  matchingObjs( MyList, Rect );
double SPOSMIN_MAT = 80.0 ; // sobreposición mínima de matching entre objetos //Mayor valor->más estricto
int INC_DEC_VIDA = 20;
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

double wtime, wtime_tot;

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

double min_op = 1.0; // overlap precision
int max_dp = 50; // distance precision

//__________________________________________________________________________________________________________________
//******************************************************************************************************************
int main()
{

        if ( USE_GPU ) {
            cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
        }

        VideoWriter output;
        output.open ( "carsTracking7.avi", CV_FOURCC('D','I','V','X'), 30, cv::Size ( 720, 480), true );

        //---------------------Información actualizada en cada cuadro------------------
        Mat frame, prev_frame, frame_vis, prev_frame_vis( 0, 0, CV_8UC3, Scalar(0,0,0) );
        Mat motion_history( h_mtn, w_mtn, CV_32FC1, Scalar(0,0,0) ); // almacena gradiente, magnitud, etc de cada pixel
        std::vector<MyList> mem_objs; // Almacena los objetos detectados y seguidos por cada cuadro.
        namedWindow("video", 1);


        //.........................Crea puntero a VideoCapture..............................
        VideoCapture *cap;
        cap = new cv::VideoCapture("../videoTest/"+name_vid);        
        if ( !cap->isOpened() ) {
            cout << "Cannot open the video file" << endl;
            return -1;
        }

        //......................... Obtiene información del video...............................
        int total_n_frms = (int)cap->get(cv::CAP_PROP_FRAME_COUNT);
        cout << "total_n_frms: " << total_n_frms <<endl;
        int fps = (int)cap->get(cv::CAP_PROP_FPS);
        cout << "fps: " << fps <<endl;


        //......................... Lee el 1er cuadro y almacenarlo como prev_frame.............
        for ( int i = 0; i < 2 ; i++ )            
            cap->read(prev_frame);
        cv::resize(prev_frame, prev_frame, Size(w, h));


        //..........................Archivar información.........................................
        ofstream fIoU, ftiempo, ffinalresults, fplotresults, fRms, fnpredbox; //fIoU para medir el número de detectados seguidos por frame

        string str_min_op = "", str_max_dp = "";
        string namefile;

        {stringstream ss; ss<<min_op;  str_min_op = ss.str();}
        {stringstream ss; ss<<max_dp;  str_max_dp = ss.str();}

        if(USE_GPU){
            ftiempo.open("tiemposParal.txt");
            fIoU.open("IoU_GPU.txt");
            namefile = "resultados_gpu_"+str_min_op+"_"+str_max_dp+"_.txt";
            const char* name1 = namefile.c_str();
            ffinalresults.open(name1);
            namefile = "plot_gpu_.txt";
            const char* name2 = namefile.c_str();
            fplotresults.open(name2, std::ios_base::app|std::ios_base::ate);
            namefile = "rms_gpu.txt";
            const char* name3 = namefile.c_str();
            fRms.open(name3);
            fnpredbox.open("n_predbox_por_Frame_gpu.txt");
        }else{
            ftiempo.open("tiemposSec.txt");
            fIoU.open("IoU_Sec.txt");
            namefile = "resultados_sec_"+str_min_op+"_"+str_max_dp+"_.txt";
            const char* name = namefile.c_str();
            ffinalresults.open(name);
            namefile = "plot_sec_.txt";
            const char* name2 = namefile.c_str();
            fplotresults.open(name2, std::ios_base::app|std::ios_base::ate);
            namefile = "rms_sec.txt";
            const char* name3 = namefile.c_str();
            fRms.open(name3);
            fnpredbox.open("n_predbox_por_Frame_sec.txt");
        }


        //..........................Para calcular las métricas..........................
        //...............................Rms, op, dp, iou..........................
        analysis analysis_ = analysis( max_dp, min_op );
        int total_boxes =  analysis_.cuentaTruthBoxes( total_n_frms - 4 );
        int acum_op = 0, acum_dp = 0;


        //..........................................................................................
        while( 1 ) {
            bool analy = false;
            c++;

                cout<<"frame: "<<c<<endl;
                ftiempo<<c<<" ";

                if ( !cap->read(frame) )     break;

                if( w != frame.cols )
                    cv::resize(frame, frame, Size(w, h));

                frame_vis = frame.clone();

                if ( (c % 1) == 0) {

                //.........................Detección de vehículos..............................
                     wtime = omp_get_wtime();
                     std::vector<Rect> det_objs;
                     function_detect_object( det_objs, frame_vis);
                     wtime = omp_get_wtime() - wtime;
                     ftiempo<<setprecision(8) << std::fixed << wtime*1000<<" ";
                     //dibujaObjs(det_objs, frame_vis, Scalar(0,255,0));
                //........................................................................................


                //.........................Detección de movimiento........................................
                    wtime = omp_get_wtime();
                    std::vector<Rect> mov_objs;
                    std::vector<double> seg_angles;
                    function_detect_motion( mov_objs, seg_angles, frame, prev_frame, motion_history);
                    wtime = omp_get_wtime() - wtime;
                    ftiempo<<setprecision(8) << std::fixed << wtime*1000<<" ";
                    //dibujaObjs(mov_objs, frame_vis, Scalar(0,0,0));
                //.......................................................................................



                //..............Control sobreposición entre detección y movimiento......................
                    wtime = omp_get_wtime();
                    std::vector<double> det_angles;
                    function_matching_motion_object(det_angles, seg_angles, det_objs, mov_objs );
                    wtime = omp_get_wtime() - wtime;
                    ftiempo<<setprecision(8) << std::fixed << wtime*1000<<" ";
                    //dibujaObjs( det_objs, frame_vis, Scalar(0,255,0));
                //.......................................................................................



                //....................Track with DLib and Memory storage........................
                    int det_objs_sz = det_objs.size();
                    int mem_objs_sz = mem_objs.size();
                    fnpredbox<<c<<" "<<det_objs_sz<<endl;
                    function_tracking(frame, frame_vis,det_angles,seg_angles,det_objs,mem_objs,
                                      fRms, fIoU, ftiempo,
                                      analy,
                                      det_objs_sz, mem_objs_sz,
                                      fnpredbox, analysis_,
                                      acum_op, acum_dp);
                //.......................................................................................


                    prev_frame = frame.clone();
                    prev_frame_vis = frame_vis;

                    if(!analy)
                        fRms<<c<<" 0"<<endl;

                    string name1= "", str1 = "";
                    {stringstream ss;    ss<<c;      str1 = ss.str();}
                    name1 = "frame_" + str1 + ".JPEG";
                    imwrite(name1, frame_vis);
                } // Fin del IF de frames intercalados


                if( prev_frame_vis.cols == 0 )
                    imshow("video", frame_vis);
                else
                    imshow("video", prev_frame_vis);

                waitKey(10);

                output.write(frame_vis);
                cout << frame_vis.rows <<endl ;
                cout << frame_vis.cols <<endl ;

        } // Fin del While


        //.........................Guarda información final................................
        double distance_precision = (double) acum_dp / (double)total_boxes;
        double overlap_precision = (double) acum_op / (double)total_boxes;
        ffinalresults<<"acum_op: "<<acum_op<<endl;
        ffinalresults<<"acum_dp: "<<acum_dp<<endl;
        ffinalresults<<"total_boxes: "<<total_boxes<<endl;
        ffinalresults<<"overlap_precision: "<<overlap_precision<<endl;
        ffinalresults<<"distance_precision: "<<distance_precision<<endl;
        fplotresults<<acum_op<<" "<<min_op<<" "<<overlap_precision<<" "<<acum_dp<<" "<<max_dp<<" "<<distance_precision<<endl;
        //.......................................................................................


        //.........................Cierra archivos........................................
        fnpredbox.close();
        ftiempo.close();
        ffinalresults.close();
        fRms.close();
        fplotresults.close();
        fIoU.close();
        //.......................................................................................


        destroyWindow("video");
        output.release();
        return 0;
}

//__________________________________________________________________________________________________________________
//******************************************************************************************************************

void dibujaObjs(std::vector<Rect> objs, Mat& img, Scalar color){
    int sz_objs = objs.size();
    for(int i = 0 ; i < sz_objs ; i++)   {
       Rect r = objs.at(i);
       cv::rectangle( img, Rect(r.x, r.y, r.width, r.height), color, 3);
    }
}

//__________________________________________________________________________________________________________________
//******************************************************************************************************************

void dibujaObj(Rect obj, Mat& img, Scalar color){
       cv::rectangle( img, Rect(obj.x, obj.y, obj.width, obj.height), color, 3);
}

//__________________________________________________________________________________________________________________
//******************************************************************************************************************
std::vector<Rect> detect1(Mat img) {
    std::vector<Rect> objs;
    std::vector<int> reject_levels;
    std::vector<double> level_weights;
    double scale_factor = 1.2;
    int min_neighbors = 8;
    cascada.detectMultiScale(img, objs, reject_levels, level_weights,
                             scale_factor, min_neighbors, 0, Size(20,20), img.size(), true);
    //if( sz_objs > 1 )
         //groupRectangles(objs, 1 , 0.2); // groupRectangles(vector<Rect>& rectList, int groupThreshold, double eps=0.2)
    return objs;
}

//__________________________________________________________________________________________________________________
//******************************************************************************************************************

std::vector<Rect> detect1_gpu(Mat img) {
    std::vector<Rect> objs;
    double scale_factor = 1.1;
    int min_neighbors = 5;
    bool findLargestObject = false; // con false el tiempo total aumenta de 44.2ms a 62.9ms
    Mat im(img.size(),CV_8UC1);

    GpuMat frame_gpu, imgBuf_gpu;
    frame_gpu.upload(img);

    if(img.channels()==3)
    {
            cv::cvtColor(img,im,CV_BGR2GRAY);
    }
    else
    {
            img.copyTo(im);
    }
    GpuMat gray_gpu(im);
    //convertAndResize(frame_gpu, gray_gpu, resized_gpu, scale_factor);
    cascade_gpu->setMinObjectSize(Size(20,20));
    //cascade_gpu->setMaxObjectSize();
    cascade_gpu->setFindLargestObject(findLargestObject);
    cascade_gpu->setScaleFactor(scale_factor);
    cascade_gpu->setMinNeighbors(min_neighbors);

    cascade_gpu->detectMultiScale(gray_gpu, imgBuf_gpu);
    cascade_gpu->convert(imgBuf_gpu, objs);

    //int detections_num = cascade_gpu.detectMultiScale(gray_gpu, imgBuf_gpu, scale_factor,min_neighbors);

    gray_gpu.release();
    imgBuf_gpu.release();

    return objs;
}
//__________________________________________________________________________________________________________________
//******************************************************************************************************************
void convertAndResize(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& gray,cv::cuda:: GpuMat& resized, double scale)
{
    if (src.channels() == 3)
    {
        cv::cuda::cvtColor( src, gray, COLOR_BGR2GRAY );
    }
    else
    {
        gray = src;
    }

    Size sz(cvRound(gray.cols * scale), cvRound(gray.rows * scale));

    if (scale != 1)
    {
        cv::cuda::resize(gray, resized, sz);
    }
    else
    {
        resized = gray;
    }
}
//__________________________________________________________________________________________________________________
//******************************************************************************************************************
bool overlapTotal(Rect r1, Rect r2) {
    int area_inter = ( r1 & r2 ).area();

    if( area_inter == r1.area() ||  area_inter == r2.area()  )
        return true;
    return false;

}


//__________________________________________________________________________________________________________________
//******************************************************************************************************************
///
/// El ratio entre el área de la intersección y el área de la región meno, debe ser
/// mayor a min_overlap
bool overlapParcial( Rect r1, Rect r2, double min_overlap ) {
    double propMenor = 0.0;
    int area_inter = ( r1 & r2 ).area();
    int res = compareD((double)r1.area(), (double)r2.area()); // Si r1 es menor entonces -1
    if(  res < 1  ) //Buscamos el menor
        propMenor = (double)area_inter/(double)r1.area();
    else
        propMenor = (double)area_inter/(double)r2.area();

    if ( (propMenor*100.0) > min_overlap )
        return true;

    return false;
}


//__________________________________________________________________________________________________________________
//******************************************************************************************************************
std::vector<Rect> movDetect(Mat frm , Mat prev_frm, int min_sz, std::vector<double> &seg_angles, Mat& motion_history) {
    std::vector<Rect> seg_bounds_mtn, seg_bounds;
    Mat motion_mask(h_mtn, w_mtn, CV_32FC1,Scalar(0,0,0));
    Mat mg_mask(h_mtn, w_mtn, CV_8UC1,Scalar(0,0,0));
    Mat mg_orient(h_mtn, w_mtn, CV_32FC1,Scalar(0,0,0));
    Mat seg_mask(h_mtn, w_mtn, CV_32FC1,Scalar(0,0,0));

    Mat frmfiltered, frame_diff, frame_diff_rz, gray_diff;
    //GaussianBlur(frm, frmfiltered, Size(5,5), 0.0);
    GaussianBlur(frm, frmfiltered, Size(5,5), 0.0);
    GaussianBlur(prev_frm, prev_frm, Size(5,5), 0.0);

    cv::absdiff(frmfiltered, prev_frm, frame_diff);
    cv::resize(frame_diff, frame_diff_rz, Size(w_mtn, h_mtn)); // resize frame_diff to get less computational cost

    cv::cvtColor(frame_diff_rz, gray_diff, CV_BGR2GRAY );
    cv::threshold(gray_diff, motion_mask, DEFAULT_THRESHOLD, 255, 0);

    double timestamp = 1000.0*clock()/CLOCKS_PER_SEC;
    cv::motempl::updateMotionHistory(motion_mask, motion_history, timestamp, MHI_DURATION);
    if(USE_GPU)
        calcMotionGradient_gpu(motion_history, mg_mask, mg_orient, MIN_TIME_DELTA, MAX_TIME_DELTA, APERTURE_SZ_SOBEL);
    else
        cv::motempl::calcMotionGradient(motion_history, mg_mask, mg_orient, MIN_TIME_DELTA, MAX_TIME_DELTA, APERTURE_SZ_SOBEL);
    cv::motempl::segmentMotion(motion_history, seg_mask, seg_bounds_mtn, timestamp, 16);


    int seg_b_size = seg_bounds_mtn.size();

    for( int i = 0; i < seg_b_size; i++ ) {
        Rect rec = seg_bounds_mtn.at(i);
        Rect ori_rec = Rect(rec.x*red_frm, rec.y*red_frm, rec.width*red_frm, rec.height*red_frm);
        if( ori_rec.area() > min_sz ) {
            seg_bounds.push_back(ori_rec);
            //seg_bounds.erase( seg_bounds.begin() + i );
            //i--;
        }else{
            //imshow("mg_orient", mg_orient);
            Mat orient_roi = mg_orient(rec);
            Mat mask_roi = mg_mask(rec);
            Mat mhi_roi = motion_history(rec);
            double angle = cv::motempl::calcGlobalOrientation(orient_roi, mask_roi, mhi_roi, timestamp, MHI_DURATION);
            seg_angles.push_back(angle);
        }
    }

    return seg_bounds;

}

//__________________________________________________________________________________________________________________
//******************************************************************************************************************
static cv::Rect dlibRectangleToOpenCV( dlib::rectangle r )
{
  return cv::Rect(cv::Point2i(r.left(), r.top()), cv::Point2i(r.right() + 1, r.bottom() + 1));
}

//__________________________________________________________________________________________________________________
//******************************************************************************************************************

static dlib::rectangle openCVRectToDlib(cv::Rect r)
{
  return dlib::rectangle((long)r.tl().x, (long)r.tl().y, (long)r.br().x - 1, (long)r.br().y - 1);
}
//__________________________________________________________________________________________________________________
//******************************************************************************************************************

void writeMatToFile(cv::Mat& m, const char* filename)
{
    ofstream fout(filename);

    if(!fout)
    {
        cout<<"File Not Opened"<<endl;  return;
    }

    for(int i=0; i<m.rows; i++)
    {
        for(int j=0; j<m.cols; j++)
        {
            fout<<m.at<float>(i,j)<<"\t";
        }
        fout<<endl;
    }

    fout.close();
}

//__________________________________________________________________________________________________________________
//******************************************************************************************************************
Point calculateMassCenter( Mat img ) {
    Mat imgThresholded, frame_gris;
    cv::cvtColor(img, frame_gris, CV_RGB2GRAY);
    cv::threshold(frame_gris, imgThresholded, 20,255,0);

    //morphological opening (removes small objects from the foreground)
    erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
    dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );

    //morphological closing (removes small holes from the foreground)
    dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
    erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );


    //Calculate the moments of the thresholded image
    Moments oMoments = moments(imgThresholded);

    double dM01 = oMoments.m01;
    double dM10 = oMoments.m10;
    double dArea = oMoments.m00;

    int posX = dM10 / dArea;
    int posY = dM01 / dArea;
    Point ctr_masa(posX, posY);

    return ctr_masa;
}
//__________________________________________________________________________________________________________________
//******************************************************************************************************************
bool matchingObjs (MyList mylist, Rect curr_rect ) {
    Rect mem_rect = mylist.box;
    // Verificar una sobreposición fuertemente exigente, con un 70% de área de sobreposición mínima
    if ( overlapTotal( mem_rect, curr_rect ) ||  overlapParcial( mem_rect, curr_rect, SPOSMIN_MAT ) )
      return true;

    return false;
}

//__________________________________________________________________________________________________________________
//******************************************************************************************************************
bool enelAmbitoGlobal(Point punto, int width, int height){
    if ( punto.x > x_lim_izq && punto.x < x_lim_der && punto.y > y_lim_sup_track && (punto.y + height) < y_lim_inf_track  )
        return true;
    return false;
}
//__________________________________________________________________________________________________________________
//******************************************************************************************************************
bool isplate(Rect& cand, Rect& obj) {
     if ( (cand.y   - (obj.y + obj.height)) < 80 && (cand.y   - (obj.y + obj.height)) > 0 && cand.x > obj.x && (cand.x + cand.width ) < obj.x + obj.width  )
         return true;
     return false;
}

//__________________________________________________________________________________________________________________
//******************************************************************************************************************
bool iscap(Rect& cand, Rect& obj) {
     if ( (obj.y - (cand.y + cand.height )) < 80  && (obj.y - (cand.y + cand.height )) > 0 && cand.x > obj.x && (cand.x + cand.width ) < obj.x + obj.width  )
         return true;
     return false;
}

//__________________________________________________________________________________________________________________
//******************************************************************************************************************
void saveThreshold(Mat frame, Rect this_rect, int c_th, int c){
    Mat this_rect_mat,  this_rect_mat_clr = frame(this_rect);
    Mat this_rect_mat_th;//(this_rect_mat.rows, this_rect_mat.cols, CV_8UC1,Scalar(0,0,0));
    cv::cvtColor(this_rect_mat_clr,this_rect_mat,CV_RGB2GRAY);
    cv::threshold(this_rect_mat, this_rect_mat_th,0,150, THRESH_OTSU );
    string name1= "", str1 = "", str2 = "";
    {stringstream ss;    ss<<c;      str1 = ss.str();}
    {stringstream ss;    ss<<c_th;      str2 = ss.str();}
    name1 = "frame_" + str1 + "_" + str2 + ".jpg";
    imwrite(name1, this_rect_mat_th);
}

//__________________________________________________________________________________________________________________
//******************************************************************************************************************
int carColorExtraction8binsHSV(Mat &img){ // Se consideran n_bins x n_bins x n_bins colores. Si n_bins=8 -> 512colores
    int n_bins = 16;

    Mat img_hsv, hue, mask, hist, backproj;
    int histSize = MAX( n_bins, 2 );
    float hranges[] = {0, 180};
    const float* phranges = hranges;

    //inRange(img_hsv, Scalar(0, 80, 10),
    //        Scalar(180, 256, 100), mask); //(hsv, lower, upper, mask)
    cv::cvtColor(img, img_hsv, CV_BGR2HSV);
    hue.create(img_hsv.rows, img_hsv.cols, CV_8UC1);
    int ch[] = { 0, 0 };
    mixChannels( &img_hsv, 1, &hue, 1, ch, 1 );
    calcHist(&hue, 1, 0, mask, hist, 1, &histSize, &phranges);
    normalize(hist, hist, 0, 255, NORM_MINMAX);
    calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
    //backproj &= mask;
    cv::resize(backproj, backproj, Size(2*backproj.rows, 2*backproj.cols));
    //imshow("BackProj", backproj);


    //int w = 400; int h = 400;
    //int bin_w = cvRound( (double) w / histSize );
    //Mat histImg = Mat::zeros( w, h, CV_8UC3 );

    float max_value = 0.0;
    int n_bins_sel =0 ;

    for( int i = 0; i < n_bins; i ++ )
    {
        float value = hist.at<float>(i);
        //log<<value<<"\t";
        //if(  value < 254.0 && value > max_value ){
        if( (compareD(value, 255.0) < 0) && value > max_value ) {
            max_value = value;
            n_bins_sel = i;
        }
    }
    return n_bins_sel;
}

//__________________________________________________________________________________________________________________
//******************************************************************************************************************
void calcMotionGradient_real( InputArray _mhi, OutputArray _mask,
                             OutputArray _orientation,
                             double delta1, double delta2,
                             int aperture_size ) // Función ORIGINAL de opencv
{
    static int runcase = 0; runcase++;

    Mat mhi = _mhi.getMat();
    Size size = mhi.size();

    _mask.create(size, CV_8U);
    _orientation.create(size, CV_32F);

    Mat mask = _mask.getMat();
    Mat orient = _orientation.getMat();

    if( aperture_size < 3 || aperture_size > 7 || (aperture_size & 1) == 0 )
        CV_Error( Error::StsOutOfRange, "aperture_size must be 3, 5 or 7" );

    if( delta1 <= 0 || delta2 <= 0 )
        CV_Error( Error::StsOutOfRange, "both delta's must be positive" );

    if( mhi.type() != CV_32FC1 )
        CV_Error( Error::StsUnsupportedFormat,
                 "MHI must be single-channel floating-point images" );

    if( orient.data == mhi.data )
    {
        _orientation.release();
        _orientation.create(size, CV_32F);
        orient = _orientation.getMat();
    }

    if( delta1 > delta2 )
        std::swap(delta1, delta2);

    float gradient_epsilon = 1e-4f * aperture_size * aperture_size;
    float min_delta = (float)delta1;
    float max_delta = (float)delta2;

    Mat dX_min, dY_max;

    // calc Dx and Dy
    Sobel( mhi, dX_min, CV_32F, 1, 0, aperture_size, 1, 0, BORDER_REPLICATE );
    Sobel( mhi, dY_max, CV_32F, 0, 1, aperture_size, 1, 0, BORDER_REPLICATE );

    int x, y;

    if( mhi.isContinuous() && orient.isContinuous() && mask.isContinuous() )
    {
        size.width *= size.height;
        size.height = 1;
    }

    // calc gradient
    for( y = 0; y < size.height; y++ )
    {
        const float* dX_min_row = dX_min.ptr<float>(y);
        const float* dY_max_row = dY_max.ptr<float>(y);
        float* orient_row = orient.ptr<float>(y);
        uchar* mask_row = mask.ptr<uchar>(y);

        cout<<"111111"<<endl;
        cv::hal::fastAtan2(dY_max_row, dX_min_row, orient_row, size.width, true);
        cout<<"22222"<<endl;

        //float f = cv::fastAtan2( dY_max_row, dX_min_row);

        // make orientation zero where the gradient is very small
        for( x = 0; x < size.width; x++ )
        {
            float dY = dY_max_row[x];
            float dX = dX_min_row[x];

            if( std::abs(dX) < gradient_epsilon && std::abs(dY) < gradient_epsilon )
            {
                mask_row[x] = (uchar)0;
                orient_row[x] = 0.f;
            }
            else
                mask_row[x] = (uchar)1;
        }
    }

    erode( mhi, dX_min, noArray(), Point(-1,-1), (aperture_size-1)/2, BORDER_REPLICATE );
    dilate( mhi, dY_max, noArray(), Point(-1,-1), (aperture_size-1)/2, BORDER_REPLICATE );

    // mask off pixels which have little motion difference in their neighborhood
    for( y = 0; y < size.height; y++ )
    {
        const float* dX_min_row = dX_min.ptr<float>(y);
        const float* dY_max_row = dY_max.ptr<float>(y);
        float* orient_row = orient.ptr<float>(y);
        uchar* mask_row = mask.ptr<uchar>(y);

        for( x = 0; x < size.width; x++ )
        {
            float d0 = dY_max_row[x] - dX_min_row[x];

            if( mask_row[x] == 0 || d0 < min_delta || max_delta < d0 )
            {
                mask_row[x] = (uchar)0;
                orient_row[x] = 0.f;
            }
        }
    }
}

//__________________________________________________________________________________________________________________
//******************************************************************************************************************
void calcMotionGradient_gpu( InputArray _mhi, OutputArray _mask,
                             OutputArray _orientation,
                             double delta1, double delta2,
                             int aperture_size )
{
    static int runcase = 0; runcase++;

    Mat mhi = _mhi.getMat();
    Size size = mhi.size();

    _mask.create(size, CV_8U);
    _orientation.create(size, CV_32F);

    Mat mask = _mask.getMat();
    Mat orient = _orientation.getMat();

    if( aperture_size < 3 || aperture_size > 7 || (aperture_size & 1) == 0 )
        CV_Error( Error::StsOutOfRange, "aperture_size must be 3, 5 or 7" );

    if( delta1 <= 0 || delta2 <= 0 )
        CV_Error( Error::StsOutOfRange, "both delta's must be positive" );

    if( mhi.type() != CV_32FC1 )
        CV_Error( Error::StsUnsupportedFormat,
                 "MHI must be single-channel floating-point images" );

    if( orient.data == mhi.data )
    {
        _orientation.release();
        _orientation.create(size, CV_32F);
        orient = _orientation.getMat();
    }

    if( delta1 > delta2 )
        std::swap(delta1, delta2);

    float gradient_epsilon = 1e-4f * aperture_size * aperture_size;
    float min_delta = (float)delta1;
    float max_delta = (float)delta2;

    Mat dX_min, dY_max;
    // calc Dx and Dy
    //Sobel( mhi, dX_min, CV_32F, 1, 0, aperture_size, 1, 0, BORDER_REPLICATE );
    //Sobel( mhi, dY_max, CV_32F, 0, 1, aperture_size, 1, 0, BORDER_REPLICATE );
    //GpuMat gray_gpu(im); //gray_gpu.release();
    cuda::GpuMat dX_min_gpu, dY_max_gpu;
    cuda::GpuMat mhi_gpu(mhi);
    Ptr<cuda::Filter> sobelFilter_dX = cv::cuda::createSobelFilter(CV_32F,CV_32F,1,0, aperture_size, 1, BORDER_REPLICATE);
    Ptr<cuda::Filter> sobelFilter_dY = cv::cuda::createSobelFilter(CV_32F,CV_32F,0,1, aperture_size, 1, BORDER_REPLICATE);
    sobelFilter_dX->apply(mhi_gpu, dX_min_gpu);
    sobelFilter_dY->apply(mhi_gpu, dY_max_gpu);

    //Mat mhi_n(mhi_gpu);
    Mat dX_min_m(dX_min_gpu);
    Mat dY_max_m(dY_max_gpu);

     dX_min = dX_min_m, dY_max = dY_max_m;

    int x, y;

    if( mhi.isContinuous() && orient.isContinuous() && mask.isContinuous() )
    {
        size.width *= size.height;
        size.height = 1;
    }

    // calc gradient
    for( y = 0; y < size.height; y++ )
    {
        const float* dX_min_row = dX_min.ptr<float>(y);
        const float* dY_max_row = dY_max.ptr<float>(y);
        float* orient_row = orient.ptr<float>(y);
        uchar* mask_row = mask.ptr<uchar>(y);

        cv::hal::fastAtan2(dY_max_row, dX_min_row, orient_row, size.width, true);

        // make orientation zero where the gradient is very small
        for( x = 0; x < size.width; x++ )
        {
            float dY = dY_max_row[x];
            float dX = dX_min_row[x];

            if( std::abs(dX) < gradient_epsilon && std::abs(dY) < gradient_epsilon )
            {
                mask_row[x] = (uchar)0;
                orient_row[x] = 0.f;
            }
            else
                mask_row[x] = (uchar)1;
        }
    }

    /*cuda::GpuMat mhi4ch;
    cuda::cvtColor(mhi_gpu, mhi4ch, COLOR_RGB2BGRA);
    std::cout<<"TYPE: " <<mhi_gpu.type()<<endl;

    const cv::Mat ker = cv::getStructuringElement( cv::MORPH_RECT, cv::Size((aperture_size-1)/2, (aperture_size-1)/2) );
    cv::Ptr<cv::cuda::Filter> erod = cv::cuda::createMorphologyFilter(CV_MOP_ERODE, mhi4ch.type(), ker);
    cv::Ptr<cv::cuda::Filter> dilat = cv::cuda::createMorphologyFilter(CV_MOP_DILATE, mhi4ch.type(), ker);
    erod->apply(mhi4ch, dX_min_gpu);
    dilat->apply(mhi4ch, dY_max_gpu);
    Mat dX_min_a(dX_min_gpu);
    Mat dY_max_a(dY_max_gpu);

     dX_min = dX_min_a, dY_max = dY_max_a;*/

    erode( mhi, dX_min, noArray(), Point(-1,-1), (aperture_size-1)/2, BORDER_REPLICATE );
    dilate( mhi, dY_max, noArray(), Point(-1,-1), (aperture_size-1)/2, BORDER_REPLICATE );

    // mask off pixels which have little motion difference in their neighborhood
    #pragma omp for private(y)
    for( y = 0; y < size.height; y++ )
    {
        const float* dX_min_row = dX_min.ptr<float>(y);
        const float* dY_max_row = dY_max.ptr<float>(y);
        float* orient_row = orient.ptr<float>(y);
        uchar* mask_row = mask.ptr<uchar>(y);

        for( x = 0; x < size.width; x++ )
        {
            float d0 = dY_max_row[x] - dX_min_row[x];

            if( mask_row[x] == 0 || d0 < min_delta || max_delta < d0 )
            {
                mask_row[x] = (uchar)0;
                orient_row[x] = 0.f;
            }
        }
    }
}


//__________________________________________________________________________________________________________________
//******************************************************************************************************************
void function_detect_motion( std::vector<Rect>& mov_objs, std::vector<double>& seg_angles, Mat& frame, Mat& prev_frame, Mat& motion_history) {

   mov_objs = movDetect(frame, prev_frame, 150, seg_angles, motion_history); //150 min_size of motion boxes
    int ii, jj ;
    for( ii = 0 ; ii < mov_objs.size() ; ii++)   {
        Rect r1 = mov_objs.at(ii);
            for( jj = ii+1 ; jj <  mov_objs.size() ; jj++)   {
                  Rect r2 = mov_objs.at(jj);
                  if( overlapTotal( r1 , r2 )  ||  overlapParcial( r1 , r2, 30.0 ) ) {
                      if ( r1.area() < r2.area() ) {
                          mov_objs.erase( mov_objs.begin() + ii );
                          seg_angles.erase( seg_angles.begin() + ii );
                          ii--;
                          jj = mov_objs.size();
                      }
                      else{
                          mov_objs.erase( mov_objs.begin() + jj );
                          seg_angles.erase( seg_angles.begin() + jj );
                          jj--;
                      }
                  }
              }
    }
}

//__________________________________________________________________________________________________________________
//******************************************************************************************************************
void function_detect_object( std::vector<Rect>& det_objs, Mat& frame_vis) {
    if ( USE_GPU ) {
         det_objs = detect1_gpu(frame_vis);
    }else {
         det_objs = detect1(frame_vis);
    }
    // Verifica overlap de cuadros detectados
           //dibujaObjs(det_objs, frame_vis, Scalar(0,0,255));
     for(int i = 0 ; i < det_objs.size() ; i++) {
         Rect r1 = det_objs.at(i);
             for(int j = i+1 ; j <  det_objs.size() ; j++)   {
                   Rect r2 = det_objs.at(j);
                   if( overlapTotal( r1 , r2 )  ||  overlapParcial( r1 , r2, 30.0 ) ){
                       if ( r1.area() < r2.area() ){
                           det_objs.erase( det_objs.begin() + i );
                           i--;
                           j=det_objs.size();
                       }
                       else{
                           det_objs.erase( det_objs.begin() + j );
                           j--;
                       }
                   }
               }
     }
}

//__________________________________________________________________________________________________________________
//******************************************************************************************************************
void function_matching_motion_object(std::vector<double>& det_angles, std::vector<double>& seg_angles, std::vector<Rect>& det_objs, std::vector<Rect>& mov_objs ) {
    int i;
    bool over;
    Rect r1;

    //#pragma omp parallel for collapse(2) private(i, over, mov_objs, det_objs, r1, r2) shared(det_angles) //num_threads(8)
    // Intento fallido de paralelización
    for(int i = 0 ; i < det_objs.size() ; i++) {
        r1 = det_objs.at(i);
        std::vector<float> angles_thisdet;
        float ave_of_angles = 0.0;
        int mov_objs_size = mov_objs.size();
        for(int j = 0 ; j <  mov_objs_size ; j++) {
          Rect r2 = mov_objs.at(j);
          if( overlapParcial( r1 , r2, 30.0 ) ) { // si esto se cumple entonces es detección de un obj en movimiento
              angles_thisdet.push_back(seg_angles.at(j));
              over = true;
          }
        }

        if( !over ) {
          det_objs.erase( det_objs.begin() + i );
          i--;
        }else{
          ave_of_angles = __gnu_parallel::accumulate(angles_thisdet.begin(), angles_thisdet.end(), 0)/angles_thisdet.size();
        }
        over = false;
        det_angles.push_back( ave_of_angles );
    }
}


//__________________________________________________________________________________________________________________
//******************************************************************************************************************
void function_tracking(Mat& frame,Mat& frame_vis,
                       std::vector<double>& det_angles,
                       std::vector<double>& seg_angles,
                       std::vector<Rect>& det_objs,
                       std::vector<MyList>& mem_objs,
                       ofstream& fRms, ofstream& fIoU, ofstream& ftiempo,
                       bool& analy,
                       int det_objs_sz,
                       int mem_objs_sz,
                       ofstream& n_predbox,
                       analysis& analysis_,
                       int acum_op, int acum_dp) {


    dlib::array2d<unsigned char> dlibImage0;
    Mat frame_gris;
    cv::cvtColor(frame, frame_gris, CV_RGB2GRAY);
    dlib::assign_image(dlibImage0, dlib::cv_image<unsigned char>(frame_gris));
    char n_case = 'n';
    if( det_objs_sz > 0 && mem_objs.empty() && c > 1)
        n_case = 'A';
    if( mem_objs_sz > 0 && det_objs_sz > 0)
        n_case = 'B';
    if ( mem_objs_sz > 0 && det_objs_sz == 0)
        n_case = 'C';
    ftiempo<<n_case<<" ";


    switch ( n_case ) {

    case 'A':
            // * I *  Almacenando en memoria los primeros objetos encontrados.
                    wtime = omp_get_wtime();
                 //#pragma omp parallel num_threads(8)
                 //{
                   // #pragma omp for schedule(static)
                    for( int i = 0 ; i < det_objs_sz ; i++) {
                        MyList obj;
                        obj.box = det_objs[i];
                        obj.angle = det_angles[i];
                        Mat myrect = frame_vis(obj.box);
                        obj.ctr_masa = calculateMassCenter( myrect ) ;
                        obj.porc_acep = 0;
                        obj.porc_vida = 100;
                        dlib::correlation_tracker tracker;
                        obj.tracker = tracker;
                        obj.tracking = false;
                        //#pragma omp critical
                        mem_objs.push_back( obj );

                    }
               // }
                    wtime = omp_get_wtime() - wtime;
                    ftiempo<<setprecision(8) << std::fixed << wtime*1000<<endl;
            break;

    case 'B':
    {
            // * II * Cuando ya se tiene objetos en memoria y se tiene objetos detectados en el frame actual

                // Matching entre objetos de memoria y objetos actualmente detectados
                wtime = omp_get_wtime();
                for( int i = 0 ; i < mem_objs_sz ; i++) {
                        bool anymatching = true;
                        MyList curr_mem_obj = mem_objs[i];
                        for( int j = 0 ; j < det_objs_sz ; j++) {
                            Rect curr_det_obj = det_objs[j];
                            if ( matchingObjs( curr_mem_obj , curr_det_obj) )  {
                                // Debería ser con quien tiene el máximo overlapping, es decir el más cercano
                                 if ( !curr_mem_obj.tracking ) {
                                     curr_mem_obj.porc_acep = curr_mem_obj.porc_acep + INC_DEC_VIDA ;
                                     curr_mem_obj.box = curr_det_obj;
                                 }
                                 anymatching = false;
                                 det_objs.erase( det_objs.begin() + j );
                                 det_objs_sz = det_objs.size();
                                 j--;
                            }
                            else{
                                  if ( isplate(curr_det_obj, curr_mem_obj.box) ||  iscap(curr_det_obj, curr_mem_obj.box) ) {
                                      cout<<"isPlate"<<endl;
                                      det_objs.erase( det_objs.begin() + j );
                                      det_objs_sz = det_objs.size();
                                      j--;
                                  }
                            }
                        }
                        if( anymatching && !curr_mem_obj.tracking ) {
                               curr_mem_obj.porc_vida = curr_mem_obj.porc_vida - INC_DEC_VIDA;
                        }
                        mem_objs[i] = curr_mem_obj;
                }

                // Crea objetos nuevos para memoria
               //#pragma omp parallel for schedule(static)
                for( int j = 0 ; j < det_objs_sz ; j++) {
                    cout << "nuevo objeto en memoria _ nuevo objeto en memoria"<<endl;
                    MyList obj; // Es nuevo objeto detectado, almacena en memoria
                    obj.box = det_objs[j];
                    obj.angle = det_angles[j];
                    Mat myrect = frame_vis(obj.box);
                    obj.ctr_masa = calculateMassCenter( myrect ) ;
                    obj.porc_acep = 0;
                    obj.porc_vida = 100;
                    dlib::correlation_tracker tracker;
                    obj.tracker = tracker;
                    obj.tracking = false;
                    //#pragma omp critical
                    mem_objs.push_back( obj );

                    mem_objs_sz = mem_objs.size();
                }


                // Continua trackings respectivos, to update trackings
                std::vector<bool> eliminar = std::vector<bool>(mem_objs_sz, false);
                int i;
                // %#pragma omp parallel num_threads(8)
                // %{
                    // %#pragma omp for private(i)
                    for( i = 0 ; i < mem_objs_sz ; i++) {
                        MyList curr_mem_obj = mem_objs[i];
                        if ( curr_mem_obj.tracking ) {
                            // verifica si el objeto está aún en escena, sino eliminarlo
                            if ( enelAmbitoGlobal(Point(curr_mem_obj.box.x, curr_mem_obj.box.y), curr_mem_obj.box.width, curr_mem_obj.box.height) ) {
                                mem_objs[i].tracker.update(dlibImage0);
                                dlib::drectangle drect = mem_objs[i].tracker.get_position();
                                cv::Rect rectt = dlibRectangleToOpenCV( drect );
                                mem_objs[i].box = rectt;
                                dibujaObj( rectt, frame_vis, Scalar(0,0,0) );
                                analy = analysis_.measure( rectt, c, acum_op, acum_dp, fRms );
                            } else {
                                eliminar[i] = true;
                            }
                        }
                    }
                // %}


                for( i = 0 ; i < mem_objs_sz ; i++)
                    if ( eliminar[i] == true ) {
                        mem_objs.erase( mem_objs.begin() + i );
                        //eliminar.erase( eliminar.begin() + i );
                        //i--;
                    }
                mem_objs_sz = mem_objs.size();
                wtime = omp_get_wtime() - wtime;
                ftiempo<<setprecision(8) << std::fixed << wtime*1000<<endl;
            break;
    }

    case 'C':
            // * III * No hay detección en el frame actual
            wtime = omp_get_wtime();
            // %#pragma omp parallel num_threads(8)
               // %{
                   // %#pragma omp for
                    for( int i = 0 ; i < mem_objs_sz ; i++) {
                        // Continua tracking con aquellos objetos en estado tracking = true.
                        if ( mem_objs[i].tracking ) {
                            mem_objs[i].tracker.update( dlibImage0 );
                            dlib::drectangle drect = mem_objs[i].tracker.get_position();
                            cv::Rect rectt = dlibRectangleToOpenCV( drect );
                            mem_objs[i].box = rectt;
                            dibujaObj( rectt, frame_vis, Scalar(0,0,0) );
                            analy =  analysis_.measure(rectt, c, acum_op, acum_dp, fRms );
                        }
                        else {
                        // Reduce porcentaje de vida de cada objeto en mem_objs
                            mem_objs[i].porc_vida =  mem_objs[i].porc_vida - INC_DEC_VIDA;
                        }
                    }
                // %}
            wtime = omp_get_wtime() - wtime;
            ftiempo<<setprecision(8) << std::fixed << wtime*1000<<endl;
            break;
    default:
            ftiempo<<0.0<<endl;
            break;

    } //F: fin del SWITCH


    // Corroborar si es necesario el isplate e isCap anteriores. o.0
    for( int i = 0 ; i < mem_objs_sz ; i++) {
        for( int j = 0 ; j < mem_objs_sz ; j++) {
            if ( isplate(mem_objs[j].box, mem_objs[i].box) ||  iscap(mem_objs[j].box, mem_objs[i].box) ) {
                mem_objs.erase( mem_objs.begin() + j );
                mem_objs_sz = mem_objs.size();
                j--;
            }
        }
    }


    // Verifica porcentajes de los objetos en memoria
    int num_objs_tracked = 0;
    for( int i = 0 ; i < mem_objs_sz ; i++) {
        MyList curr_mem_obj = mem_objs[i];
        if ( curr_mem_obj.tracking )
            num_objs_tracked++;

        if ( curr_mem_obj.porc_acep >= 100 && !curr_mem_obj.tracking ){
            Rect this_rect = curr_mem_obj.box;
            this_rect = Rect(this_rect.x + (this_rect.x*0.05),
                             this_rect.y,
                             this_rect.width-(this_rect.width*0.1),
                             this_rect.height);

            curr_mem_obj.tracker.start_track( dlibImage0, centered_rect( openCVRectToDlib(this_rect), this_rect.width, this_rect.height ));
            curr_mem_obj.tracking = true;
            curr_mem_obj.box = this_rect;
            mem_objs[i] = curr_mem_obj;
        }
        if ( curr_mem_obj.porc_vida <= 0 ) {
            mem_objs.erase( mem_objs.begin() + i );
            mem_objs_sz = mem_objs.size();
            i--;
        }
    }

} // Fin de la función tracking

