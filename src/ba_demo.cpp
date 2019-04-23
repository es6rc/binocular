#include <iostream>
// for opencv 
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <boost/concept_check.hpp>
// for g2o
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/types/slam3d/se3quat.h>
#include <g2o/types/sba/types_six_dof_expmap.h>


using namespace std;
using namespace cv;

int findCorrespondingPoints( const cv::Mat& img1, const cv::Mat& img2, vector<cv::Point2f>& points1, vector<cv::Point2f>& points2 );

// Camera Intrinsic Parameters
double cx = 325.5;
double cy = 253.5;
double fx = 518.0;
double fy = 519.0;

int main( int argc, char** argv )
{
    // Usaage：./ba_example [1st image] [2nd image]
    if (argc != 3)
    {
        cout<<"Usage: ba_example img1, img2"<<endl;
        exit(1);
    }
    
    // read in imagas
    cv::Mat img1 = cv::imread( argv[1] ); 
    cv::Mat img2 = cv::imread( argv[2] ); 
    
    // find matching points

    vector<cv::Point2f> pts1, pts2;
    if ( findCorrespondingPoints( img1, img2, pts1, pts2 ) == false )
    {
        cout<<"Not enough matching points"<<endl;
        return 0;
    }
    cout<<"Found "<<pts1.size()<<"matching pairs"<<endl;
    // Build graph in g2o
    // Build optimizer and solver first
    g2o::SparseOptimizer optimizer;
    // Use Cholmod Linear Solver
    // 6*3 : 6 for # of params in pose vector, 3 for # of params in map points vector
    std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver;
    linearSolver = 
        g2o::make_unique<g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>>();    
    // L-M  
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver)) );

    optimizer.setAlgorithm( solver );
    optimizer.setVerbose( false );
    
    // Add two pose point (vertex)
    for ( int i=0; i<2; i++ )
    {
        g2o::VertexSE3Expmap* v = new g2o::VertexSE3Expmap();
        v->setId(i);
        if ( i == 0)
            v->setFixed( true ); // Set the first fixed point to zero
        // Set Pose to unit vector, because no info about pose is provided
        v->setEstimate( g2o::SE3Quat() );
        optimizer.addVertex( v );
    }
    // Set the first frame as referrence
    for ( size_t i=0; i<pts1.size(); i++ )
    {
        g2o::VertexSBAPointXYZ* v = new g2o::VertexSBAPointXYZ();
        v->setId( 2 + i );
        // Set the depth to 1
        double z = 1;
        double x = ( pts1[i].x - cx ) * z / fx; 
        double y = ( pts1[i].y - cy ) * z / fy; 
        v->setMarginalized(true);
        v->setEstimate( Eigen::Vector3d(x,y,z) );
        optimizer.addVertex( v );
    }
    
    // Add Cmera Params
    g2o::CameraParameters* camera = new g2o::CameraParameters( fx, Eigen::Vector2d(cx, cy), 0 );
    camera->setId(0);
    optimizer.addParameter( camera );
    
    // Initialize Edge
    // First Frame 
    vector<g2o::EdgeProjectXYZ2UV*> edges;
    for ( size_t i=0; i<pts1.size(); i++ )
    {
        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
        edge->setVertex( 0, dynamic_cast<g2o::VertexSBAPointXYZ*>   (optimizer.vertex(i+2)) );
        edge->setVertex( 1, dynamic_cast<g2o::VertexSE3Expmap*>     (optimizer.vertex(0)) );
        edge->setMeasurement( Eigen::Vector2d(pts1[i].x, pts1[i].y ) );
        edge->setInformation( Eigen::Matrix2d::Identity() );
        edge->setParameterId(0, 0);
        // Kernel fn.
        edge->setRobustKernel( new g2o::RobustKernelHuber() );
        optimizer.addEdge( edge );
        edges.push_back(edge);
    }
    // second image
    for ( size_t i=0; i<pts2.size(); i++ )
    {
        g2o::EdgeProjectXYZ2UV*  edge = new g2o::EdgeProjectXYZ2UV();
        edge->setVertex( 0, dynamic_cast<g2o::VertexSBAPointXYZ*> (optimizer.vertex(i+2)) );
        edge->setVertex( 1, dynamic_cast<g2o::VertexSE3Expmap*>   (optimizer.vertex(1)) );
        edge->setMeasurement( Eigen::Vector2d(pts2[i].x, pts2[i].y ) );
        edge->setInformation( Eigen::Matrix2d::Identity() );
        edge->setParameterId(0,0);
        // Kernel fn.
        edge->setRobustKernel( new g2o::RobustKernelHuber() );
        optimizer.addEdge( edge );
        edges.push_back(edge);
    }
    
    cout<<"Start optimization"<<endl;
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    cout<<"End optimization"<<endl;
    
    // More interested in homogenous transformation
    g2o::VertexSE3Expmap* v = dynamic_cast<g2o::VertexSE3Expmap*>( optimizer.vertex(1) );
    Eigen::Isometry3d pose = v->estimate();
    cout<<"Pose="<<endl<<pose.matrix()<<endl;

    // and all keypoints locations
    for ( size_t i=0; i<pts1.size(); i++ )
    {
        g2o::VertexSBAPointXYZ* v = dynamic_cast<g2o::VertexSBAPointXYZ*> (optimizer.vertex(i+2));
        cout<<"vertex id "<<i+2<<", pos = ";
        Eigen::Vector3d pos = v->estimate();
        cout<<pos(0)<<","<<pos(1)<<","<<pos(2)<<endl;
    }

    // Estimate # of Inliers 
    int inliers = 0;
    for ( auto e:edges )
    {
        e->computeError();
        // chi2 = error*\Omega*error, if chi2 is huge, this edge doesn't match with others
        if ( e->chi2() > 1 )
        {
            cout<<"error = "<<e->chi2()<<endl;
        }
        else 
        {
            inliers++;
        }
    }
    
    cout<<"inliers in total points: "<<inliers<<"/"<<pts1.size()+pts2.size()<<endl;
    optimizer.save("ba.g2o");
    return 0;
}


int findCorrespondingPoints( const cv::Mat& img1, const cv::Mat& img2, vector<cv::Point2f>& points1, vector<cv::Point2f>& points2 )
{
    Ptr<ORB> orb = ORB::create();
    // orb->setMaxFeatures();
    
    vector<KeyPoint> kp1, kp2;
    Mat desp1, desp2;
    orb->detectAndCompute(img1, cv::Mat(), kp1, desp1);
    orb->detectAndCompute(img2, cv::Mat(), kp2, desp2);
    cout<<"Individually found "<<kp1.size()<<" and "<<kp2.size()<<" match points"<<endl;
    
    Mat skp1, skp2;
    drawKeypoints(img1, kp1, skp1);
    drawKeypoints(img2, kp2, skp2);
    imshow("Keypoints1", skp1);
    imshow("Keypoints2", skp2);
    waitKey(0);

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    double knn_match_ratio=0.8;
    vector<vector<cv::DMatch>> matches_knn;
    matcher->knnMatch( desp1, desp2, matches_knn, 2 );
    vector<DMatch> matches;
    for ( size_t i=0; i<matches_knn.size(); i++ )
    {
        if (matches_knn[i][0].distance < knn_match_ratio * matches_knn[i][1].distance )
            matches.push_back( matches_knn[i][0] );
    }
    
    if (matches.size() <= 20) // Too few match pairs
        return false;
    
    for ( auto m:matches )
    {
        points1.push_back( kp1[m.queryIdx].pt );
        points2.push_back( kp2[m.trainIdx].pt );
    }
    
    return true;
}

