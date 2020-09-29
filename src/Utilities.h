/*******************************************************************************
 * Copyright (C) 2020 Optimal Synthesis Inc
 *
 * This file is a part of Monte Carlo simulation for fire control schedulers
 *
 * All Rights Reserved.
 ******************************************************************************/

// Author: Parikshit Dutta
#ifndef _UTILITIES_H
#define _UTILITIES_H

#include <vector>
#include <string>
#include <random>
#include <memory>
#include <iostream>
#include <exception>
#include <type_traits>
#include <cassert>
#include <set>
#include <algorithm>
#include <cfloat>
#include <eigen3/Eigen/Dense>
#include <chrono>

//Multivariate Normal Matrix vector multiplication
static double getNormalDistributionValue(const std::vector<double>& muvec, const std::vector<std::vector<double> >& Sig){
    assert(Sig.size() == Sig[0].size());
    assert(muvec.size() == Sig.size());
    int ndim = muvec.size();
    //First Create Eigen from std vector
    Eigen::VectorXd mu = Eigen::VectorXd::Map(muvec.data(), ndim);
    
    Eigen::MatrixXd sig(ndim,ndim);
    for (int i = 0; i < ndim; i++)
        sig.row(i) = Eigen::VectorXd::Map(&Sig[i][0],ndim);

    Eigen::MatrixXd var = sig*sig;

    double expval = mu.transpose() * (var.inverse() ) * mu;
    double detvar =  var.determinant();
    double nprob = pow(2*M_PI,-ndim/2.0) * pow(detvar,-0.5) * exp(-0.5*expval);
    return nprob;
}

/* For multivariate Normal samples 
We first convert the covariance matrix into a diagonal matrix by EV decomposition. Then each of the transformed dimensions are i.i.d. Then sample using regular normal distribution for each dimension and then transform back to original correlated version.  
Let 
Var = A* Lam * A^-1

1/(2*pi)^(k/2) *(det(Var))^(1/2) * exp(-0.5*(x-\mu)^T * Var^-1 * (x-\mu))
= 1/(2*pi)^(k/2) *(det(Var))^(1/2) * exp(-0.5*(x-\mu)^T * A* Lam^-1 *A^-1 * (x-\mu))
A^T = A^-1
Let y = A^T(x-mu)
= 1/(2*pi)^(k/2) *(det(Lam))^(1/2) * exp(-0.5*y^T * Lam^-1 * y )
Prod of EV = Determinant
x = Ay + mu;
 */


static std::vector<double> genMultiVariateNormalSample(const std::vector<double>& muvec, const std::vector<std::vector<double> >& Sig, std::mt19937 eng){
    assert(Sig.size() == Sig[0].size());
    assert(muvec.size() == Sig.size());
    int ndim = muvec.size();
    //First Create Eigen from std vector
    Eigen::VectorXd mu = Eigen::VectorXd::Map(muvec.data(), ndim);
    
    Eigen::MatrixXd sig(ndim,ndim);
    for (int i = 0; i < ndim; i++)
        sig.row(i) = Eigen::VectorXd::Map(&Sig[i][0],ndim);

    Eigen::MatrixXd var = sig*sig;
    
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> esolver(var);
    Eigen::VectorXd evals = esolver.eigenvalues().real();
    Eigen::MatrixXd evecs = esolver.eigenvectors().real();

    Eigen::VectorXd samp(ndim);


    for(int i = 0;i<ndim;++i){
        std::normal_distribution ndist(0.0,sqrt(evals(i)) );//this takes in mean and std dev
        samp(i) = ndist(eng);
    }
    // std::cout << "In normal distribution \n";
    // std::cout <<" mu = " << mu << std::endl << " sig = " << sig << std::endl;
    // std::cout << "evecs = " << evecs << std::endl << " evals = " << evals << std::endl;
    // std::cout << "bfore samp = " << samp << std::endl;    
    samp = evecs*samp + mu;
    // std::cout << "after samp = " << samp << std::endl;
    // std::cout <<"out of normal.\n";
    std::vector<double> nsamp(ndim);
    Eigen::VectorXd::Map(&nsamp[0],ndim) = samp;

    return nsamp;
}


//Find out if something is a vector
template<typename T> struct is_vector : public std::false_type {};

template<typename T, typename A>
struct is_vector<std::vector<T, A>> : public std::true_type {};

template< class T >
inline constexpr bool is_vector_v = is_vector<T>::value;
//end here

//Find out if something is a vector of vectors
template<typename T> struct is_vector_vector : public std::false_type {};

template<typename T, typename A>
struct is_vector_vector< std::vector < std::vector<T, A> > > : public std::true_type {};

template< class T >
inline constexpr bool is_vector_vector_v = is_vector_vector<T>::value;
//end here


template<class T>
class Distribution{
public:
    virtual T getRandomPoint() = 0;
    virtual T getRandomPoint(const T& x) = 0;
    virtual double evalCondDensity(const T& pt,const T& cpt) = 0;
    virtual double evalDensity(const T& pt) = 0;
    
    //Follow rule of five else unique_ptr produces memory leak
    virtual ~Distribution<T>(){} 
    Distribution<T>() = default;
    Distribution<T>(const Distribution<T>& that) = default;
    virtual Distribution<T>& operator=(const Distribution<T>& that) = default;
    Distribution<T>(Distribution<T>&& that) = default;
    virtual Distribution<T>& operator=(Distribution<T>&& that) = default;  
};
/*
Normal Distribution obtained from 
https://en.wikipedia.org/wiki/Normal_distribution
Custom wrapper for the STL class std::normal_distribution
*/
template<class T,class P>
class NormalDist: public Distribution<T>{
    const T _mu; const P _sig;
public:
    NormalDist(const T& mu,const P& sig):_mu(mu),_sig(sig){}

    T getRandomPoint(const T& x) override{
        T pt;
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::mt19937 eng(seed);
        
        if constexpr (is_vector_v<T> && is_vector_vector_v<P>){
            return genMultiVariateNormalSample(x,_sig,eng);
        }
        else if constexpr (std::is_floating_point_v<T>){
            std::normal_distribution ndist(x,_sig);
            return ndist(eng);
        }
        else{ //we can do set
            throw std::invalid_argument("Can only use scalar or vectors");
        }
        return pt;
    }

    T getRandomPoint() override{
        T pt;
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::mt19937 eng(seed);
        
        if constexpr (is_vector_v<T> && is_vector_vector_v<P>){
            return genMultiVariateNormalSample(_mu,_sig,eng);
        }
        else if constexpr (std::is_floating_point_v<T>){
            std::normal_distribution ndist(_mu,_sig);
            return ndist(eng);
        }
        else{ //we can do set
            throw std::invalid_argument("Can only use scalar or vectors");
        }
        return pt;
    }

    //Conditional Normal density. Just center around the point
    //Square the Sig and invert it.
    double evalCondDensity(const T& pt,const T& cpt) override{
        double val = 0.0;
        if constexpr (is_vector_v<T> && is_vector_vector_v<P>){
            T dpt;

            auto itp = pt.begin(), itc = cpt.begin();
            for(;itp != pt.end() && itc != cpt.end(); ++itp, ++itc ){
                dpt.push_back(*itp - *itc);
            }
            val = getNormalDistributionValue(dpt,_sig);
        }
        else if constexpr (std::is_floating_point_v<T>){
            val = 1.0/(sqrt(2.0*M_PI)*_sig) *exp( - 0.5*pow( (pt-cpt)/_sig,2.0) );
        }
        else{ //we can do set
            throw std::invalid_argument("Can only use scalar or vectors");
        }
        return val;
    }

    double evalDensity(const T& pt){
        return evalCondDensity(pt,_mu); 
    }

};
/*
Rayleigh Distribution obtained from 
https://en.wikipedia.org/wiki/Rayleigh_distribution
Did not get an example for multivariate distribution. Assumed each axis is independent. 
*/
template<class T>
class Rayleigh: public Distribution<T>{
    const T _sp; // It is same for all directions.
public:
    static constexpr double _mean_scale{sqrt(2.0/M_PI)};
    Rayleigh(const T& sp):_sp(sp){}
    /*For Rayleigh distribution we cannot draw from the distribution as it does not fall under distributions that are coverd under the STL library. So just return the mean
    NOT USED MUCH
    Input: None
    Output: A random sample point
    */
    T getRandomPoint(){return _sp;};
    /*For Rayleigh distribution we cannot draw from the distribution as it does not fall under distributions that are coverd under the STL library. So just return the mean.
    NOT USED MUCH
    Input: Mean of distribution
    Output: A random point. 
    */
    T getRandomPoint(const T& x){return x;};

    /*
    The MH Algorithm evalues conditional density for only proposal (Kernel distribution).
    Hence not important.
    Input : evaluation point x, mean of distribution
    Output: PDF value.
    */
    double evalCondDensity(const T& pt,const T& cpt) override{
        double val = 1.0;
        if constexpr (is_vector_v<T>){
            //can be any distribution as no formal type exists
            assert(pt.size() == cpt.size());            
            T sp;
            std::transform(cpt.begin(), cpt.end(), std::back_inserter(sp),[](const auto& v){
                return v*Rayleigh::_mean_scale;
            });
            int k = 0;
            for(;k<pt.size(); ++k){         
                if (pt[k] > 0.0)       
                    val = val*(pt[k])/(sp[k]*sp[k])*exp(- (pt[k]*pt[k])/(2*sp[k]*sp[k] ) );
                else {
                    val = 0.0; break;
                }
            }
        }
        else if constexpr (std::is_floating_point_v<T>){
            if (pt <= 0) val = 0.0;
            else{
                T sp = cpt*Rayleigh::_mean_scale;
                val = val*(pt)/(sp*sp)*exp(- (pt*pt)/(2*sp*sp ) );
            }

        }
        else {
            try{
                assert(pt.size() == cpt.size());
                T sp;
                std::transform(cpt.begin(), cpt.end(), std::inserter( sp,sp.begin() ),[](const auto& v){
                    return v*Rayleigh::_mean_scale;
                    });
                
                auto siter = sp.begin();
                auto piter = pt.begin();
                while(siter != sp.end() && piter != pt.end()){
                    if (*piter <= 0) {
                        val = 0; 
                        break;
                        }
                    val = val*( *piter)/( (*siter)*(*siter) )*exp(- ( (*piter) * (*piter) )/(2*(*siter)*(*siter) ) );
                    std::advance(siter,1); std::advance(piter,1);
                }
            }
            catch (std::exception& e){
                std::cout <<"An exception occured. " << e.what() << std::endl;
            }
            
        }
        return val;
    }
    /*
    This evaluates Rayleigh density. This is to be used in MH.   
    We are scaling the Rayleigh parameter and passing it to the conditional density function.
    */
    double evalDensity(const T& pt) override{
        if constexpr (is_vector_v<T>){
            T sp;
            std::transform(_sp.begin(), _sp.end(), std::back_inserter(sp),[](const auto& v){
                return v/Rayleigh::_mean_scale;
            });
            return evalCondDensity(pt,sp);
        }
        else if constexpr (std::is_floating_point_v<T>){            
            return evalCondDensity(pt,_sp/(Rayleigh::_mean_scale) );            
        }
        else{
            try{
                T sp;
                std::transform(_sp.begin(), _sp.end(),std::inserter(sp,sp.begin()),[](const auto& b){
                    return b/(Rayleigh::_mean_scale);
                    });
                return evalCondDensity(pt,sp);
            }
            catch (std::exception& e){
                std::cout <<"An exception occured. " << e.what() << std::endl;
            }
        }

        return 0.0;
    }
};

/*
Uniform distribution
PDF = h \forall x1, x2, ..., xn
\int_D h dx1dx2......dxn = 1
=> h = 1/\int_D dx1....dxn
=>h = 1/V, where V is the volume in n-d hyperspace or Area in 2-D and length in 1D
Can be tricky. It can be a hyperspace polygon or a hypersphere.
Vol of hypersphere:
 {\displaystyle V_{n}(R)={\frac {\pi ^{\frac {n}{2}}}{\Gamma \left({\frac {n}{2}}+1\right)}}x_1x_2\ldotsx_n,}
  Γ(n) = (n − 1)! 
  Γ(n + 1/2) = (n − 1/2) · (n − 3/2) · … · 1/2 · π1/2 
Vol of hyperpolygon:
  V = x1.x2.x3....xn
*/
enum class Voltype {Line,Sphere, Polygon};
template<class T>
class UniformDist: public Distribution<T>{    
    const T _dims;
    const Voltype _type;
    double _pdf;
    const T _center;
public:    
    UniformDist(const T& dims, const T& center, const Voltype& type = Voltype::Line):_dims(dims),_center(center),_type(type){
        if constexpr (std::is_floating_point_v<T>){            
            _pdf = 1.0/_dims;            
        }            
        else if constexpr (is_vector_v<T>){
            int n = _dims.size();
            double K = 1;
            if (_type == Voltype::Sphere){
                K = pow(M_PI,n/2.0)/std::tgamma(n/2.0+1.0);
                for( const auto& v:_dims){                
                    K *= (v/2.0);
                }
            }
            else{
                for( const auto& v:_dims){                
                    K *= v;
                }
            }
            _pdf = K!=0.0?1.0/K:DBL_MAX;            
        }
        else{
             throw std::invalid_argument("Can only use scalar or vectors");
         }
            
    }




    T getRandomPoint(const T& x){
        return x;
    }
    T getRandomPoint(){ 
         if constexpr (is_vector_v<T>){
             T res(_dims.size());
             return res;
         }        
         else if constexpr (std::is_floating_point_v<T>){
             return 0.0;
         }
         else{
             throw std::invalid_argument("Can only use scalar or vectors");
         }
    }
    //Used here if the mean is centered around cpt. Does not matter here as all is same for uniform dist. 
    double evalCondDensity(const T& pt,const T& cpt){
        if constexpr (is_vector_v<T>){
            //Cant be 1 d
            int n = pt.size();
            if (_type == Voltype::Polygon){
                for(int i=0;i<n;++i){
                    if (pt[i]> cpt[i] + _dims[i]/2.0 || pt[i] < cpt[i]-_dims[i]/2.0) return 0.0;                
                }   
            }
            else if (_type == Voltype::Sphere){
                double val = 0;

                for(int i=0;i<n; ++i){
                    val += ( pow(pt[i]-cpt[i],2)/pow(_dims[i]/2.0,2) );                    
                }
                if (val > 1.0) return 0;
            }
         
        }
        else {
            if (pt > cpt + _dims/2.0 || pt <cpt -_dims/2.0) return 0.0;            
        }
        return _pdf;
    }
    double evalDensity(const T& pt){            
        return evalCondDensity(pt,_center);    
    }

};


template<class T>
class Sampling{
public:
    virtual std::vector<T> getSamples(const int& N) = 0;
    virtual void initBurnOut() = 0;
    virtual T getSingleSample() = 0;
    virtual ~Sampling<T>(){} 
    Sampling<T>() = default;
    Sampling<T>(const Sampling<T>& that) = default;
    virtual Sampling<T>& operator=(const Sampling<T>& that) = default;
    Sampling<T>(Sampling<T>&& that) = default;
    virtual Sampling<T>& operator=(Sampling<T>&& that) = default;  
};

/*
 From : Bayesian Inference: Metropolis-Hastings Sampling, Ilker Yildirim, Department of Brain and Cognitive Sciences, University of Rochester, Rochester, NY 14627.
  http://www.mit.edu/~ilkery/papers/MetropolisHastingsSampling.pdf
*/
template<class T>
class MetropolisHastings:public Sampling<T>{

    const int _burn_out;
    std::unique_ptr<Distribution<T> > _targ_dist;
    std::unique_ptr<Distribution<T> > _prop_dist;

    //Just for generating a point
    T _x;

    bool MHAlgoCycle(T& x0, std::uniform_real_distribution<double>& udist, std::mt19937& eng){
        T xcand = _prop_dist->getRandomPoint(x0);
        // std::for_each(xcand.begin(), xcand.end(), [](const auto& v){ std::cout << v <<" ";});
        // std::cout<< std::endl;
        double num = _prop_dist->evalCondDensity(x0,xcand)*_targ_dist->evalDensity(xcand);
        double den = _prop_dist->evalCondDensity(xcand,x0)*_targ_dist->evalDensity(x0);        
        
        double alp = 1.0;
        if (den > 0.0){    
            alp = std::min(alp,(1.0*num)/den );                
        }
                
        double val =  udist(eng);
        // std::cout << "num = " << num <<" den = " << den << " alp = " << alp << " val = " << val << std::endl;
        
        if (alp > val ){            
            x0 = xcand;
            return true;
        }
        return false;

    }

public:
    MetropolisHastings(const int& b_out, std::unique_ptr<Distribution<T> > targ_dist, std::unique_ptr<Distribution<T> > prop_dist): _burn_out(b_out), _targ_dist(std::move(targ_dist)),_prop_dist(std::move(prop_dist)){        
    }

    std::vector<T> getSamples(const int& N) override{
        //Initialize random device for uniform distribution
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::mt19937 eng(seed);
        std::uniform_real_distribution<double> udist(0.0,1.0);

        //Get output container
        std::vector<T> op;

        //the first step is to generate a value randomly from target distribution. 
        T x0 = _prop_dist->getRandomPoint();

        //First burnout
        int cycles = 0;
        while(cycles++ < _burn_out){
            MHAlgoCycle(x0,udist,eng);
        }

        //Once the burnout is done then push back into vector
        while(op.size() < N){
            if(MHAlgoCycle(x0,udist,eng)){
                op.push_back(x0);
            }
        }
        return op;
    }
     
    //Only for MH algo.
    void initBurnOut(){
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::mt19937 eng(seed);
        std::uniform_real_distribution<double> udist(0.0,1.0);

        //Get output container
        std::vector<T> op;

        //the first step is to generate a value randomly from target distribution. 
        _x = _prop_dist->getRandomPoint();

        //First burnout
        int cycles = 0;
        while(cycles++ < _burn_out){
            MHAlgoCycle(_x,udist,eng);
        }

    }

    T getSingleSample(){
        unsigned  seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::mt19937 eng(seed);
        std::uniform_real_distribution<double> udist(0.0,1.0);

        while(MHAlgoCycle(_x,udist,eng) == false){}
        return _x;
    }
};

/*
Similar to MH algo with a little bit of difference. Go to 
https://en.wikipedia.org/wiki/Rejection_sampling for specifics.
*/
template<class T>
class AcceptReject:public Sampling<T>{    
    const int _M; //Expected number of iterations needed. Typically higher the better. 
    std::unique_ptr<Distribution<T> > _targ_dist;
    std::unique_ptr<Distribution<T> > _prop_dist;
public:
    AcceptReject(const int& M, std::unique_ptr<Distribution<T> > td, std::unique_ptr<Distribution<T> > pd):_M(M),_targ_dist(std::move(td)), _prop_dist(std::move(pd)){}

    std::vector<T> getSamples(const int& N) override {
        std::vector<T> op;
        
        //Initialize random device for uniform distribution
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::mt19937 eng(seed);
        std::uniform_real_distribution<double> udist(0.0,1.0);

        while(op.size() < N){
            T x0 = _prop_dist->getRandomPoint();
            // std::cout << x0 << std::endl;
            double num = _targ_dist->evalDensity(x0);
            double den = _prop_dist->evalDensity(x0);   
            double u = udist(eng);
            // std::cout << "u = " << u << " num = " << num <<" den = " << den  << " num/(_M*den) = " << num/(_M*den) << std::endl;
            // std::cout << "size of op = " << op.size() << std::endl;
                     
            if (u < num/(_M*den) ){
                op.push_back(x0);
            }

        }

        return op;
    }

    void initBurnOut() override {}

    T getSingleSample() override {
        int nsamp = 1;
        std::vector<T> samp = getSamples(nsamp);
        return samp[0];
    }


};

#endif