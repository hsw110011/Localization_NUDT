#include "ParticleScoring.h"
#include "ParticleFilter.h"
#include "Motion2D.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void Close(double actual, double expected, double tolerance=1e-6) {
    if (std::abs(actual-expected)>=tolerance)
        throw std::runtime_error("expected "+std::to_string(expected)+", actual "+std::to_string(actual));
}
torch::Tensor Probabilities(const torch::Device& device) {
    auto obs=torch::tensor({.7f,.2f,.2f,.3f,.1f,.5f,0.f,.8f}).reshape({1,4,1,2}).to(device);
    return torch::cat({obs,torch::ones({1,1,1,2},obs.options())},1);
}
void CheckRoutes(const torch::Device& device) {
    using sem_loc::ScoreMode;
    auto obs=Probabilities(device);
    auto probability=sem_loc::ObservationWeights(obs,ScoreMode::NoEntropy);
    auto changed=obs.clone();changed.slice(1,3,4).fill_(1);
    Require(torch::equal(probability,sem_loc::ObservationWeights(changed,ScoreMode::NoEntropy)),"no-entropy must ignore H");
    auto entropy=sem_loc::ObservationWeights(obs,ScoreMode::Entropy);
    Require(torch::allclose(entropy.select(3,0),probability.select(3,0)),"H=0 preserves contribution");
    Require(torch::allclose(entropy.select(3,1),probability.select(3,1)*.2),"linear confidence must be 1-H");
    Require(sem_loc::ObservationWeights(changed,ScoreMode::Entropy).sum().item<float>()==0,"H=1 must contribute zero");
    changed.slice(1,3,4).fill_(.5);
    auto half=sem_loc::ObservationWeights(changed,ScoreMode::Entropy);
    auto masks=torch::tensor({1.f,0.f,0.f,1.f,0.f,0.f}).reshape({1,3,1,2}).to(device);
    masks=torch::cat({masks,torch::ones({1,1,1,2},masks.options())},1);
    auto baseline=sem_loc::ObservationWeights(masks,ScoreMode::Baseline);
    auto tdf=(torch::arange(1,13,torch::kFloat32)/12).reshape({1,3,2,2}).to(device);
    auto grids=torch::tensor({-1.f,-1.f,1.f,-1.f,-1.f,1.f,1.f,1.f}).reshape({2,1,2,2}).to(device);
    auto costs=sem_loc::ComputeParticleCosts(tdf,grids,baseline);
    Close(costs[0].item<double>(),7./12);Close(costs[1].item<double>(),11./12);
    costs=sem_loc::ComputeParticleCosts(tdf,grids,probability);
    Close(costs[0].item<double>(),9.8/12);Close(costs[1].item<double>(),13.8/12);
    auto entropy_costs=sem_loc::ComputeParticleCosts(tdf,grids,entropy);
    Close(entropy_costs[0].item<double>(),4.04/12);Close(entropy_costs[1].item<double>(),6.44/12);
    Require(torch::allclose(sem_loc::ComputeParticleCosts(tdf,grids,half),costs*.5),"uniform confidence must not cancel in a denominator");
    auto doubled=sem_loc::ComputeParticleCosts(tdf,torch::cat({grids,grids},2),torch::cat({probability,probability},3));
    Require(torch::allclose(doubled,costs*2),"twice the observations must give twice the score");
    for(auto mode:{ScoreMode::NoEntropy,ScoreMode::Entropy}) {
        auto masked=obs.clone();masked.slice(1,4,5).zero_();
        auto weights=sem_loc::ObservationWeights(masked,mode);
        Require(sem_loc::ComputeParticleCosts(tdf,grids,weights).sum().item<float>()==0,"unobserved cells contributed to the score");
    }
    masks.slice(1,3,4).zero_();
    Require(sem_loc::ObservationWeights(masks,ScoreMode::Baseline).sum().item<float>()==0,"baseline visibility mask ignored");
    std::cout << device << " analytic linear scores, channel matching, mask, confidence and observation count: OK\n";
}
void CheckSampling() {
    using sem_loc::ScoreMode;
    auto tdf=(torch::arange(1,13,torch::kFloat32)/12).reshape({1,3,2,2});
    auto hwc=tdf.squeeze(0).permute({1,2,0}).contiguous();
    cv::Mat map(2,2,CV_32FC3,hwc.data_ptr<float>());
    auto grids=torch::tensor({-1.f,-1.f,1.f,-1.f,-1.f,1.f,1.f,1.f}).reshape({2,1,2,2});
    auto poses=torch::tensor({1.,0.,90.,1.,1.,90.},torch::kFloat64).reshape({2,3});
    auto obs=Probabilities(torch::kCPU);
    auto masks=torch::tensor({1.f,0.f,0.f,1.f,0.f,0.f}).reshape({1,3,1,2});
    masks=torch::cat({masks,torch::ones({1,1,1,2})},1);
    for(auto mode:{ScoreMode::Baseline,ScoreMode::NoEntropy,ScoreMode::Entropy}) {
        auto weights=sem_loc::ObservationWeights(mode==ScoreMode::Baseline?masks:obs,mode);
        auto batch=sem_loc::ComputeParticleCosts(tdf,grids,weights);
        auto loop=sem_loc::ComputeForloopParticleCosts(map,poses,weights);
        Require(torch::allclose(batch,loop,1e-5,1e-6),"loop/batch differ at integer coordinates");
        if(torch::cuda::is_available())
            Require(torch::allclose(batch,sem_loc::ComputeParticleCosts(tdf.cuda(),grids.cuda(),weights.cuda()).cpu(),1e-5,1e-6),"CUDA/batch differ");
    }
    // Strict OOB includes coordinates just beyond the edge, not only far outside.
    auto edge_tdf=torch::tensor({0.f,.2f,.4f,.6f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f}).reshape({1,3,2,2});
    auto edge_hwc=edge_tdf.squeeze(0).permute({1,2,0}).contiguous();
    cv::Mat edge_map(2,2,CV_32FC3,edge_hwc.data_ptr<float>());
    auto edge_grid=torch::tensor({-1.f,-1.f,-1.02f,-1.f,1.02f,0.f,-1.f,1.02f,1.f,1.f,-5.f,-5.f}).reshape({6,1,1,2});
    auto edge_poses=torch::empty({6,3},torch::kFloat64);
    for(int i=0;i<6;++i) {
        edge_poses[i][0]=(edge_grid[i][0][0][0].item<double>()+1)/2;
        edge_poses[i][1]=(edge_grid[i][0][0][1].item<double>()+1)/2;
        edge_poses[i][2]=90.;
    }
    auto weight=torch::tensor({1.f,0.f,0.f}).reshape({1,3,1,1});
    auto expected=torch::tensor({0.f,1.f,1.f,1.f,.6f,1.f});
    Require(torch::allclose(sem_loc::ComputeParticleCosts(edge_tdf,edge_grid,weight),expected),"GPU-style OOB penalty is not exactly one");
    Require(torch::allclose(sem_loc::ComputeForloopParticleCosts(edge_map,edge_poses,weight),expected),"CPU OOB penalty is not exactly one");
    if(torch::cuda::is_available())
        Require(torch::allclose(sem_loc::ComputeParticleCosts(edge_tdf.cuda(),edge_grid.cuda(),weight.cuda()).cpu(),expected),"CUDA OOB penalty differs");
    // Subpixel interior interpolation: OpenCV uses a 1/32 interpolation table.
    auto inner_grid=torch::tensor({-.46f,-.38f,.4f,.6f}).reshape({2,1,1,2});
    auto inner_pose=torch::tensor({.27,.31,90.,.7,.8,90.},torch::kFloat64).reshape({2,3});
    Require(torch::allclose(sem_loc::ComputeParticleCosts(edge_tdf,inner_grid,weight),sem_loc::ComputeForloopParticleCosts(edge_map,inner_pose,weight),0,.01),"subpixel interpolation exceeds tolerance");
    std::cout << "CPU/CUDA sampling, near/far OOB and subpixel coordinates: OK\n";
}
void CheckPosterior() {
    SatelliteData map;
    map.tdf_map=cv::Mat(128,128,CV_32FC3);map.satellite_map=cv::Mat::zeros(128,128,CV_8UC3);
    map.geo_bounds={113.,28.00023,113.00026,28.};map.zone_num=49;map.resolution=.2;
    map.origin_x=696667.1231450655;map.origin_y=3098839.7406396237;
    for(int r=0;r<128;++r)for(int c=0;c<128;++c)
        map.tdf_map.at<cv::Vec3f>(r,c)=cv::Vec3f(std::min(c,100),std::min(r,100),std::min(.3*c+.2*r,100.));
    CoordConverter converter(map);
    cv::Mat normalized;map.tdf_map.convertTo(normalized,CV_32F,.01);
    for(auto backend:{sem_loc::ComputeBackend::Gpu,sem_loc::ComputeBackend::Forloop})
    for(auto mode:{sem_loc::ScoreMode::Baseline,sem_loc::ScoreMode::NoEntropy,sem_loc::ScoreMode::Entropy}) {
        std::cout << "Posterior backend=" << int(backend) << " mode=" << int(mode) << std::endl;
        sem_loc::Config config;config.backend=backend;config.score_mode=mode;config.tdf_value_scale=100.;
        torch::manual_seed(123);
        ParticleFilter pf(2,config);pf.Init(map,map.origin_x+12,map.origin_y-12,90.,1.,.5,0.);
        auto states=pf.GetRawParticles();auto poses=torch::empty({2,3},torch::kFloat64);
        for(int i=0;i<2;++i) {
            // The active scorer maps UTM -> WGS84 -> the NPZ's geographic pixel grid.
            auto blh=converter.gauss_to_wgs84(states[i].x,states[i].y);
            auto pixel=converter.wgs84_to_pixel(blh.Lon,blh.Lat);
            poses[i][0]=pixel.x;poses[i][1]=pixel.y;poses[i][2]=states[i].theta;
        }
        cv::Mat observation=cv::Mat::zeros(400,400,mode==sem_loc::ScoreMode::Baseline?CV_8UC4:CV_32FC(5));
        if(mode==sem_loc::ScoreMode::Baseline) {
            observation.at<cv::Vec4b>(200,200)=cv::Vec4b(1,0,0,1);
            observation.at<cv::Vec4b>(201,200)=cv::Vec4b(0,1,0,1);
        } else {
            observation.at<cv::Vec<float,5>>(200,200)=cv::Vec<float,5>(.7f,.2f,.1f,.9f,1.f);
            observation.at<cv::Vec<float,5>>(201,200)=cv::Vec<float,5>(.2f,.3f,.5f,.9f,1.f);
        }
        cv::Mat f;observation.convertTo(f,CV_32F);
        auto obs_tensor=torch::from_blob(f.data,{400,400,f.channels()},torch::kFloat32).permute({2,0,1}).unsqueeze(0);
        auto weights=sem_loc::ObservationWeights(obs_tensor,mode);
        if(mode==sem_loc::ScoreMode::Entropy)Require(weights.sum().item<float>()<1,"low-confidence regression setup is wrong");
        auto score=sem_loc::ComputeForloopParticleCosts(normalized,poses,weights).to(torch::kFloat64);
        for(int update=0;update<2;++update) {
            auto prior=torch::tensor({states[0].weight,states[1].weight},torch::kFloat64);
            auto log_post=torch::log(prior.clamp_min(1e-12))-10.*score;
            auto expected=torch::softmax(log_post,0);
            pf.Update(observation,10.);
            states=pf.GetRawParticles();
            Close(states[0].weight,expected[0].item<double>(),.003);
            Close(states[0].weight+states[1].weight,1.);
        }
        Require(std::abs(states[0].weight-.5)>1e-5,"positive low-confidence observations were silently skipped");
        cv::Mat no_observation=cv::Mat::zeros(400,400,observation.type());
        pf.Update(no_observation,10.);auto unchanged=pf.GetRawParticles();
        Require(states[0].weight==unchanged[0].weight,"empty visibility mask changed prior");
        if(mode==sem_loc::ScoreMode::Entropy) {
            for(int r:{200,201})observation.at<cv::Vec<float,5>>(r,200)[3]=1.f;
            pf.Update(observation,10.);unchanged=pf.GetRawParticles();
            Require(states[0].weight==unchanged[0].weight,"max-entropy observations changed prior");
            for(int r:{200,201})observation.at<cv::Vec<float,5>>(r,200)[3]=0.f;
        }
        pf.Update(observation,1e6);auto stable=pf.GetRawParticles();
        Require(std::isfinite(stable[0].weight)&&std::isfinite(stable[1].weight),"large negative log likelihood was unstable");
        Close(stable[0].weight+stable[1].weight,1.);
    }
    std::cout << "Full Update: raw-map normalization, 3 methods x 2 backends, Bayesian prior, low confidence, no evidence and large-score stability: OK\n";
}
int main() {
    torch::NoGradGuard guard;
    CheckRoutes(torch::kCPU);
    if(torch::cuda::is_available())CheckRoutes(torch::kCUDA);
    CheckSampling();CheckPosterior();
}
