/*******************************************************************************
 * Copyright (C) 2020 Parikshit Dutta
 *
 * All Rights Reserved.
 ******************************************************************************/
// Author: Parikshit Dutta

#pragma once

#include "gtest/gtest.h"
#include "Utilities.h"
#include <iostream>
#include <cmath>
#include <list>

/*
TESTS RANDOM NUMBER GENERATORS USED IN THE PROJECT.
*/

class PDFTest:  public ::testing::Test{
    std::unique_ptr<Distribution< double > > _dptr;    
    std::unique_ptr<Distribution<std::vector<double> > > _vdptr;    
    
public:
    double getRayleighDist(const double& sparam,const double& val){
        _dptr = std::make_unique<Rayleigh<double> >(sparam);
        return _dptr->evalDensity(val);
    }    

    double getRayleighDist(const std::vector<double>& sparam,const std::vector<double>& val){
        _vdptr = std::make_unique<Rayleigh<std::vector<double> > >(sparam);
        return _vdptr->evalDensity(val);
    }    

    double getNormalDist(const double& mu, const double& sig, const double& val){
        _dptr = std::make_unique<NormalDist<double,double> >(mu,sig);
        return _dptr->evalDensity(val);   
    }

    double getNormalDist(const std::vector<double>& mu, const std::vector<std::vector<double> >& sig, const std::vector<double>& val){
        _vdptr = std::make_unique<NormalDist< std::vector<double>,std::vector<std::vector<double> > > >(mu,sig);
        return _vdptr->evalDensity(val);   
    }  


    double getUniformDist(const double& l, const double& r, const double& val) {
        assert(r > l);
        double dim = r-l;
        double mid = l +(r-l)/2.0;        
        _dptr = std::make_unique<UniformDist<double> >(dim,mid);
        return _dptr->evalCondDensity(val,mid);
    }

    double getUniformDist(const std::vector<double>& v1, const std::vector<double>& v2, const std::vector<double>& val, const Voltype& v) {
        std::vector<double> dim;
        std::vector<double> mid;
        if (v == Voltype::Polygon){
            std::transform(v2.begin(), v2.end(),v1.begin(), std::back_inserter(dim),std::minus<double>());

        
            std::transform(v2.begin(), v2.end(),v1.begin(), std::back_inserter(mid),[](const double& a, const double& b){
                return (a+b)/2.0;
            } );
                
        }
        else{ //Sphere
            dim = v1, mid = v2;
        }
        _vdptr = std::make_unique<UniformDist<std::vector<double> > >(dim,mid,v);
        return _vdptr->evalCondDensity(val,mid);
    }
};

TEST_F(PDFTest,RayleighDistributionDBL){
    EXPECT_FLOAT_EQ(getRayleighDist(1.0,1.0),0.60653065971);
    EXPECT_FLOAT_EQ(getRayleighDist(2.0,1.0),0.22062422564);
    EXPECT_FLOAT_EQ(getRayleighDist(2.0,0.5),0.12115415431);
    EXPECT_FLOAT_EQ(getRayleighDist(0.5,1.5),0.06665397922);
}

TEST_F(PDFTest,RayleighDistributionVDBL){
    EXPECT_FLOAT_EQ(getRayleighDist({1.0},{1.0}),0.60653065971);    
    EXPECT_FLOAT_EQ(getRayleighDist({2.0},{0.5}),0.12115415431);
    EXPECT_FLOAT_EQ(getRayleighDist({2.0,1.0},{0.5,1.0}),0.07348370914);
}

TEST_F(PDFTest,NormalDistributionDBL){
    EXPECT_FLOAT_EQ(getNormalDist(0,1.0,99.9),0.0);    
    EXPECT_FLOAT_EQ(getNormalDist(2,3.0,99.9),0.0);    
    EXPECT_FLOAT_EQ(getNormalDist(0,1.0,2.0),0.05399096651);    
    EXPECT_FLOAT_EQ(getNormalDist(5.0,10.0,4.0),0.03969525474);    
    EXPECT_FLOAT_EQ(getNormalDist(0,1.0,2.0),getNormalDist(0,1.0,-2.0));    
    EXPECT_FLOAT_EQ(getNormalDist(5.0,10.0,4.0),getNormalDist(5.0,10.0,6.0));   
}   


TEST_F(PDFTest,NormalDistributionVDBL){
    EXPECT_FLOAT_EQ(getNormalDist({0.0,0.0},{ {1.0,0.0}, {0.0,1.0} },{0.0,0.0}), 0.15915494309189535);
    EXPECT_FLOAT_EQ(getNormalDist({0.0,0.0},{ {1.0,0.0}, {0.0,1.0} },{1.0,0.0}), 0.09653235263005393);
}

TEST_F(PDFTest,UniformDistributionDBL){
    EXPECT_FLOAT_EQ(getUniformDist(0,1.0,99.9),0.0);    
    EXPECT_FLOAT_EQ(getUniformDist(-10,10.0,3),1.0/(10.0+10.0));    
    EXPECT_FLOAT_EQ(getUniformDist(-10,10.0,10.0),1.0/(10.0+10.0));    
    EXPECT_FLOAT_EQ(getUniformDist(-10,10.0,-10.0),1.0/(10.0+10.0));    
    EXPECT_FLOAT_EQ(getUniformDist(-10,10.0,-10.000000001),0);    
}  

TEST_F(PDFTest,UniformDistributionVDBL){
    // 2-D
    EXPECT_FLOAT_EQ(getUniformDist({0,-10},{1.0,10},{99.9,0}, Voltype::Polygon),0.0);            
    EXPECT_FLOAT_EQ(getUniformDist({-5,-10},{5.0,10.0},{2.0,2.0},Voltype::Polygon),1.0/(10.0+10.0)*1.0/(5.0+5.0));    
    EXPECT_FLOAT_EQ(getUniformDist({-5,-10},{5.0,10.0},{5.0,10.0},Voltype::Polygon),1.0/(10.0+10.0)*1.0/(5.0+5.0));    
    EXPECT_FLOAT_EQ(getUniformDist({-5,-10},{5.0,10.0},{5.0,10.00000000001},Voltype::Polygon),0.0);    
    EXPECT_FLOAT_EQ(getUniformDist({2.0,2.0},{0.0,0.0},{0,0}, Voltype::Sphere),1.0/(M_PI));            
    EXPECT_FLOAT_EQ(getUniformDist({2.0,2.0},{0.0,0.0},{0,1.0}, Voltype::Sphere),1.0/(M_PI));            
    EXPECT_FLOAT_EQ(getUniformDist({2.0,2.0},{0.0,0.0},{2.0,2.0}, Voltype::Sphere),0.0);            
    EXPECT_FLOAT_EQ(getUniformDist({1.0,2.0},{0.0,0.0},{0.5,0.0}, Voltype::Sphere),1.0/(M_PI*0.5));            

    // 3-D
    EXPECT_FLOAT_EQ(getUniformDist({-5,-10,-10.0},{5.0,10.0,30.0},{5.0,10.0,5.0},Voltype::Polygon),1.0/(10.0+10.0)*1.0/(5.0+5.0)*1.0/(30+10.0));    
    EXPECT_FLOAT_EQ(getUniformDist({-5,-10,-5.0},{5.0,10.0,30.0},{5.0,10.0,-10.0},Voltype::Polygon),0.0);    
    EXPECT_FLOAT_EQ(getUniformDist({-5,-10,-5.0},{5.0,10.0,30.0},{5.0,10.0,-5.0},Voltype::Polygon),1.0/(10.0+10.0)*1.0/(5.0+5.0)*1.0/(30+5.0));    

    EXPECT_FLOAT_EQ(getUniformDist({2.0,2.0,2.0},{0.0,0.0,0.0},{0,0,0}, Voltype::Sphere),3.0/(4.0*M_PI));            
    EXPECT_FLOAT_EQ(getUniformDist({2.0,2.0,2.0},{0.0,0.0,0.0},{1.0,2.0,1.0}, Voltype::Sphere),0.0);            
    EXPECT_FLOAT_EQ(getUniformDist({1.0,2.0,2.0},{0.0,0.0,0.0},{0.5,0.0,0.0}, Voltype::Sphere),3.0/(4.0*M_PI*0.5));            

}

/* Scalar values */
template<typename T>
class SamplingTestScalar:  public ::testing::Test{

    std::unique_ptr<Sampling<double> > _sdptr; //Distribution is univariate
public:
    std::vector<double> getSamples(const int& N,const int& K, const double& sparam,const double& mu, const double& sig){
        std::unique_ptr<Distribution<double> > tdist = std::make_unique<Rayleigh<double> >(sparam);
        std::unique_ptr<Distribution<double> > pdist = std::make_unique<NormalDist<double,double> >(mu,sig);

        _sdptr = std::make_unique<T>(K, std::move(tdist),std::move(pdist));

        return _sdptr->getSamples(N);
    }
    void initBurnOut(const int& K, const double& sparam,const double& mu, const double& sig){
        
        std::unique_ptr<Distribution<double> > tdist = std::make_unique<Rayleigh<double> >(sparam);
        std::unique_ptr<Distribution<double> > pdist = std::make_unique<NormalDist<double,double> >(mu,sig);

        _sdptr = std::make_unique<T>(K, std::move(tdist),std::move(pdist));
        _sdptr->initBurnOut();
    }

    double getSample(){
        return _sdptr->getSingleSample();
    }
};

using SamplingMethods = ::testing::Types<MetropolisHastings<double>,AcceptReject<double> >;
TYPED_TEST_SUITE(SamplingTestScalar, SamplingMethods);

TYPED_TEST(SamplingTestScalar,SampleFromRayleighScalar){
    int nsamp =  100000;
    int bout = 100;
    double scalef = 1.0;
    double prop_m = 1.0;
    double prop_s = 1.0;

    std::vector<double> samp = this->getSamples(nsamp,bout,scalef,prop_m,prop_s);
    double v = std::accumulate(samp.begin(),samp.end(), 0.0);
    double mv = v/samp.size();
    double varv = std::accumulate(samp.begin(), samp.end(), 0.0, [&mv](double val, const double& v){
            return std::move(val) + pow( (v-mv),2);
    });

    varv = varv/(nsamp-1);
    EXPECT_NEAR(mv,scalef*sqrt(M_PI/2.0),0.05);
    EXPECT_NEAR(varv,( (4.0-M_PI)/2.0 )*pow(scalef,2),1.0e-2 );
}

/*First burnout then sampling*/
TYPED_TEST(SamplingTestScalar,SampleFromRayleighScalarSingleSampleDraw){
    int nsamp =  100000;
    int bout = 100;
    double scalef = 1.0;
    double prop_m = 1.0;
    double prop_s = 1.0;
    
    this->initBurnOut(bout,scalef,prop_m,prop_s);
    std::vector<double> samp;
    for(int i=0;i<nsamp; ++i) samp.push_back( this->getSample() );

    double v = std::accumulate(samp.begin(),samp.end(), 0.0);
    double mv = v/samp.size();
    double varv = std::accumulate(samp.begin(), samp.end(), 0.0, [&mv](double val, const double& v){
            return std::move(val) + pow( (v-mv),2);
    });

    varv = varv/(nsamp-1);
    EXPECT_NEAR(mv,scalef*sqrt(M_PI/2.0),0.05);
    EXPECT_NEAR(varv,( (4.0-M_PI)/2.0 )*pow(scalef,2),0.05 );
}

/* Vector values */
template<typename T>
class SamplingTestVector:  public ::testing::Test{
    std::unique_ptr<Sampling<std::vector<double> > > _svdptr; //Distribution is multivariate
public:
    
    std::vector<std::vector<double> > getSamples(const int& N,const int& K, const std::vector<double>& dims, const Voltype& v, const std::vector<double>& mu, const std::vector<std::vector<double> >& sig){
        std::unique_ptr<Distribution<std::vector<double> > > tdist = std::make_unique<UniformDist<std::vector<double> > >(dims, mu, v);
        std::unique_ptr<Distribution<std::vector<double> > > pdist = std::make_unique<NormalDist< std::vector<double>,std::vector<std::vector<double> > > >(mu,sig);
        _svdptr = std::make_unique<T>(K, std::move(tdist),std::move(pdist));

        return _svdptr->getSamples(N);
    }


};

