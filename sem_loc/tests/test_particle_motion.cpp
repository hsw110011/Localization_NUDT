#include "ParticleFilter.h"
#include "Motion2D.h"
#include <iostream>
#include <stdexcept>
#include <algorithm>

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
double angleError(double a, double b) {
    return std::abs(motion2d::Normalize((a-b)*motion2d::kDegToRad))*motion2d::kRadToDeg;
}
SatelliteData makeMap() {
    SatelliteData map;
    map.tdf_map = cv::Mat(128,128,CV_32FC3,cv::Scalar(.1f,.1f,.1f));
    map.satellite_map = cv::Mat::zeros(128,128,CV_8UC3);
    map.geo_bounds = {113.,28.00023,113.00026,28.};
    map.zone_num=49; map.resolution=.2;
    map.origin_x=696667.1231450655; map.origin_y=3098839.7406396237;
    return map;
}
void checkMotion() {
    auto map=makeMap();
    const double x=map.origin_x+12., y=map.origin_y-12.;
    sem_loc::Config config; config.backend=sem_loc::ComputeBackend::Forloop;
    torch::manual_seed(42);
    ParticleFilter pf(2048,config);
    pf.Init(map,x,y,70.,0.,0.,0.);
    for(const auto& p:pf.GetRawParticles()) {
        require(p.x==x && p.y==y,"UTM initialization lost precision");
        require(angleError(p.theta,70.)<1e-9,"zero yaw standard deviation ignored");
    }
    pf.Predict(0.,0.,10.);
    auto rotated=pf.GetRawParticles();
    double yaw_variance=0.;
    for(const auto& p:rotated) {
        require(p.x==x && p.y==y,"pure rotation added XY jitter");
        yaw_variance+=(p.theta-80.)*(p.theta-80.)/rotated.size();
    }
    require(angleError(pf.EstimatePose()[2],80.)<.2,"relative turn mean is incorrect");
    require(yaw_variance>1.,"relative turn did not retain yaw uncertainty");
    pf.Predict(.01,0.,20.);
    auto moved=pf.GetRawParticles();
    double dx=0.,dy=0.,expected_x=0.,expected_y=0.;int tiny=0;
    for(size_t i=0;i<moved.size();++i) {
        dx+=(moved[i].x-x)/moved.size();dy+=(moved[i].y-y)/moved.size();
        const auto expected=motion2d::Rotate(.01,0.,rotated[i].theta*motion2d::kDegToRad);
        expected_x+=expected.x/moved.size();expected_y+=expected.y/moved.size();
        if(std::abs(moved[i].y-y)>1e-8 && std::abs(moved[i].y-y)<.1) ++tiny;
    }
    require(tiny>2000,"centimeter UTM motion was quantized");
    require(std::hypot(dx-expected_x,dy-expected_y)<.002,"translation used the current yaw");
    pf.Predict(0.,0.,0.);
    auto stationary=pf.GetRawParticles();
    for(size_t i=0;i<moved.size();++i)
        require(stationary[i].x==moved[i].x && stationary[i].y==moved[i].y && stationary[i].theta==moved[i].theta,
                "stationary odometry changed particles");

    pf.Init(map,x,y,0.,0.,0.,30.);
    auto before=pf.GetRawParticles();
    pf.Predict(1.,.3,90.);
    auto after=pf.GetRawParticles();
    dx=dy=expected_x=expected_y=0.;
    for(size_t i=0;i<after.size();++i) {
        const auto expected=motion2d::Rotate(1.,.3,before[i].theta*motion2d::kDegToRad);
        dx+=(after[i].x-before[i].x)/after.size();dy+=(after[i].y-before[i].y)/after.size();
        expected_x+=expected.x/after.size();expected_y+=expected.y/after.size();
    }
    require(std::hypot(dx-expected_x,dy-expected_y)<.01,"translation did not use each particle's previous yaw");
    pf.Resample();
    auto resampled=pf.GetRawParticles();
    auto extrema=std::minmax_element(resampled.begin(),resampled.end(),[](const auto& a,const auto& b){return a.theta<b.theta;});
    require(extrema.second->theta-extrema.first->theta>30.,"resampling overwrote yaw diversity");
    pf.Init(map,x,y,179.,0.,0.,4.);
    require(angleError(pf.EstimatePose()[2],179.)<.3,"weighted circular mean failed near +/-180 degrees");
    for(const auto& p:pf.GetRawParticles())require(p.theta>=-180. && p.theta<180.,"initial yaw was not wrapped");
}
void checkEntropyCorrectsYaw() {
    auto map=makeMap();
    const double x=map.origin_x+12., y=map.origin_y-12.;
    CoordConverter converter(map);
    const auto center=converter.gauss_to_pixel(x,y);
    for(int r=0;r<128;++r)for(int c=0;c<128;++c)
        map.tdf_map.at<cv::Vec3f>(r,c)=cv::Vec3f(std::abs(r-center.y-30.f),std::abs(c-center.x-20.f),0.f);
    cv::Mat observation=cv::Mat::zeros(400,400,CV_32FC(5));
    observation.at<cv::Vec<float,5>>(200,230)=cv::Vec<float,5>(1.f,0.f,0.f,0.f,1.f);
    observation.at<cv::Vec<float,5>>(220,200)=cv::Vec<float,5>(0.f,1.f,0.f,.9f,1.f);
    observation.at<cv::Vec<float,5>>(220,201)=cv::Vec<float,5>(0.f,1.f,0.f,.9f,1.f);
    std::vector<ParticleFilter::ParticleState> posterior[2];double headings[2];
    for(int i=0;i<2;++i) {
        sem_loc::Config config;config.backend=sem_loc::ComputeBackend::Forloop;
        config.score_mode=i ? sem_loc::ScoreMode::Entropy : sem_loc::ScoreMode::NoEntropy;
        torch::manual_seed(1111);
        ParticleFilter pf(128,config);
        pf.Init(map,x,y,70.,0.,0.,30.);
        pf.Update(observation,500.);
        posterior[i]=pf.GetRawParticles();headings[i]=pf.EstimatePose()[2];
        double sum=0.;for(const auto& p:posterior[i])sum+=p.weight;
        require(std::abs(sum-1.)<1e-6,"posterior weights do not normalize");
    }
    double max_difference=0.;
    for(size_t i=0;i<posterior[0].size();++i)
        max_difference=std::max(max_difference,std::abs(posterior[0][i].weight-posterior[1][i].weight));
    require(max_difference>1e-3,"entropy did not affect the actual PF posterior");
    require(headings[0]-headings[1]>5.,"entropy-weighted matching failed to correct particle heading");
    std::cout << "Actual Update: no-entropy yaw=" << headings[0] << ", entropy yaw=" << headings[1]
              << ", max posterior difference=" << max_difference << '\n';
}
int main() {
    checkMotion();checkEntropyCorrectsYaw();
    std::cout << "3-DOF motion, yaw uncertainty, UTM precision, circular mean and entropy correction: OK\n";
}
