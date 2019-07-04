#include "validation.h"
#include "compareD.h"



analysis::analysis(int dp, double op)
{
    this->threshold_dp  = dp;
    this->threshold_op = op;
    this->prev_nfrm = -1;
    this->acum_sqr = 0.0;
    this->rms_prev_frm = 0.0;
    this->prev_num = -1;
    //this->frms.open("rms_per_frame.txt");
}

void analysis::parserLabels() {
    int nfrms = 4;
    string str1 = "", namefile = " ";
    ifstream in;

    for ( int i = 0; i < nfrms ; i++ ) {
       {stringstream ss; ss<<i+1;  str1 = ss.str(); }
        namefile = "../labels/frame_"+str1+".txt";
        //namefile = "test3.txt";
        const char* name = namefile.c_str();
        in.open(name);
        if ( ! in ) {
           cout << "Error: Can't open the file"<<name<<".\n";
           exit(1);
        }
        int num = -1, x = -1, y = -1, width = -1, height = -1; 1;
        int ant_num=-1;
        in >> num;
        cout<<"num: "<<num<<endl;
            if ( num != 0 ){
                for(int j=0 ; j<num ; j++ ) {
                    in >> x;
                    cout<<"x: "<<x<<endl;
                    in >> y;
                    cout<<"y: "<<y<<endl;
                    in >> width;
                    cout<<"width: "<<width<<endl;
                    in >> height;
                    cout<<"height: "<<height<<endl;
                }
            }
            else{
                cout<<"No hay nada"<<endl;
            }
            cv::Rect r = cv::Rect( x, y, width, height );
        in.close();
    }
}

bool analysis::measure(cv::Rect rect_pred, int n_frm,  int& acum_op, int& acum_dp, ofstream& frms){

    string str1 = "", namefile = " ";
    ifstream in;
    {stringstream ss; ss<<n_frm;  str1 = ss.str(); }
     namefile = "../labels/frame_"+str1+".txt";
     const char* name = namefile.c_str();
     in.open(name);
     if ( ! in ) {
         cout<<"No existe "<<namefile<<endl;
        return false;
     }

     int num = -1, x = -1, y = -1, x2 = -1, y2 = -1;
     in >> num;
     if ( num == 0 && n_frm > 70 ) {
     }else{
        if ( num != 0 ) {
         double min_error2 = 10000;
         double min_error = this->threshold_dp + 1;
         cv::Rect rect_grou, rect_grou_closer;
         for(int j = 0 ; j < num ; j++ ) {
             in >> x;
             in >> y;
             in >> x2;
             in >> y2;
             rect_grou = cv::Rect( x, y, x2-x, y2-y );
             Point2f cntr_pred = calculateRectCenter(rect_pred);
             Point2f cntr_grou = calculateRectCenter(rect_grou);

             double error_dist = euclideanDist(cntr_pred, cntr_grou);

             if (  error_dist < min_error ) {
                 min_error = error_dist;
             }
             if( error_dist < min_error2){
                 rect_grou_closer = rect_grou;
                min_error2 = error_dist;
             }

         }
         if( min_error <= this->threshold_dp)
             acum_dp++;

         double iou = calculateIoU( rect_pred, rect_grou_closer );
         if( iou >= this->threshold_op)
             acum_op++;

         //Para cálculo de RMS
         if( n_frm != this->prev_nfrm ){
             this->rms_prev_frm = sqrt(this->acum_sqr);
                 cout<<"rms_prev_frm:"<<rms_prev_frm<<" prev_num:"<<prev_num<<endl;

             this->rms_prev_frm = this->rms_prev_frm/(double)this->prev_num;
                 cout<<"rms_prev_frm:"<<rms_prev_frm<<endl;

             this->acum_sqr = 0.0;
             frms<<prev_nfrm<<" "<<this->rms_prev_frm<<endl;
             this->prev_nfrm = n_frm;
         }
         this->acum_sqr = this->acum_sqr + pow(min_error,2.0);
     }
        this->prev_num = num;
     }
     in.close();
     return true;
}


bool analysis::verifyMassCenters(Point2f& p, Point2f& q) {
    double max_accep = 20.0;
    if( euclideanDist(p, q) > max_accep  ){

    }
    else{

    }
}


double analysis::calculateIoU( Rect r1, Rect r2) {
    double area_inter = ( r1 & r2 ).area();
    double area_union = ( r1 | r2 ).area();
    double iou = area_inter / area_union; // 0-1

    return iou;
}


bool analysis::overlapParcial( Rect r1, Rect r2, double min_overlap ) {
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


Point2f analysis::calculateMassCenter( Mat roi ) {
    Mat imgThresholded, gray_diff;
    cv::cvtColor(roi, gray_diff, CV_BGR2GRAY );
    threshold(gray_diff, imgThresholded, 20,255,0);

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

    //calculate the position
    Point2f ctr;
    ctr.x = dM10 / dArea;
    ctr.y = dM01 / dArea;

    return ctr;
}

int analysis::cuentaTruthBoxes( int total_n_frms ) {
    string str1 = "", namefile = " ";
    int total_boxes = 0, num;
    ifstream in;
    for( int n_frm = 0; n_frm < total_n_frms; n_frm++ ){
        {stringstream ss; ss<<n_frm;  str1 = ss.str(); }
         namefile = "../labels/frame_"+str1+".txt";
         const char* name = namefile.c_str();
         in.open(name);
         if ( in ) {
             in >> num;
             total_boxes = total_boxes + num;
         }
         in.close();
    }
    return total_boxes;
}

double analysis::euclideanDist(Point2f p, Point2f q) {
    Point2f diff = p - q;
    return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
}
Point2f analysis::calculateRectCenter( cv::Rect boundRect ) {
    return Point2f( (boundRect.x + (boundRect.width/2)), (boundRect.y + (boundRect.height/2)) );
}

void analysis::closeFile_frms(){
    //this->frms.close();
}
