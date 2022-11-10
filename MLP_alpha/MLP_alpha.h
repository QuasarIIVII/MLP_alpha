//#pragma once
#ifndef __MLP_alpha__
#define __MLP_alpha__
#include<iostream>
#include<iomanip>
#include<math.h>
#include<time.h>
#include<sys/time.h>
#include<stdlib.h>
long double randl(long double min, long double max){
	using namespace std;
	cout<<max-min<<endl;
	static unsigned int a;
	struct timeval tv;
	gettimeofday(&tv,NULL);
	srand((long long)tv.tv_sec*1000000+tv.tv_usec+a&0xFFFFFFFF);
	return (long double)((long long)(a=rand())*RAND_MAX*RAND_MAX+rand()*RAND_MAX+rand())*(max-min)/RAND_MAX/RAND_MAX+min;
}
long double linear(long double x){return x;}
long double _dtanh(long double x){return 2/(coshl(x*2)+1);}
namespace MLPa{
	template<class _rInt, class _rFloat>
	class MLPa;
}

namespace SLPa{
	using namespace std;
	template<class _rInt, class _rFloat>
	class SLPa{
		friend class MLPa::MLPa<_rInt,_rFloat>;
		private:struct{
			_rInt weight;
		}num;
		private:_rFloat*weight=0;
		private:typedef struct _sS0{
			_rFloat (*activation)(_rFloat);
			_rFloat (*dactivation)(_rFloat);
		}_sS0;
		private:_sS0 funcs;
		
		//Default Constructor
		public:SLPa(){
			;//std::cout<<"C0"<<std::endl;
			weight=0;
			num.weight=-1;
		}
		//Normal Constructor
		public:SLPa(_rInt numOfWeights){
			;//std::cout<<'A'<<numOfWeights<<endl;
			;//std::cout<<"CA"<<std::endl;
			this->num.weight=numOfWeights;
			this->weight=new _rFloat[numOfWeights+1];
			for(_rInt i=numOfWeights;i+1;i--)weight[i]=randl(0.0L,10.0L), cout<<"w : "<<weight[i]<<endl;
			this->funcs.activation=&tanhl; // ========Need to handle NullPtrException========
			this->funcs.dactivation=&_dtanh;
			//weight[num.weight] -> bias
		}
		//Destructor
		public:~SLPa(){
			;//std::cout<<"~"<<endl;
			delete[] weight;
		}
		//Copy Constructor
		SLPa(const SLPa& src){
			;//std::cout<<'B'<<src.num.weight<<endl;
			;//std::cout<<"CB"<<std::endl;
			this->num.weight=src.num.weight;
			this->weight=new _rFloat[this->num.weight+1];
			for(_rInt i=this->num.weight;i+1;i--)this->weight[i]=src.weight[i];
		}
		//Assignment Operator
		SLPa& operator=(const SLPa& src){
			;//std::cout<<'C'<<src.num.weight<<endl;
			;//std::cout<<"CC"<<std::endl;
			delete[] weight;
			this->num.weight=src.num.weight;
			this->weight=new _rFloat[this->num.weight+1];
			for(_rInt i=this->num.weight;i+1;i--)this->weight[i]=src.weight[i];
			return *this;
		}
		
		//Member functions
		public:_rFloat calc(_rFloat*data){
//			cout<<data[0]<<endl;
			cout<<"SLP Calc : "<<num.weight<<endl;
			_rFloat s=0;
			_rInt i;
			for(i=0; i<num.weight; i++)s+=data[i]*weight[i];
			s+=weight[i];
			return funcs.activation(s);
		}
		private:_rFloat calcDeAct(_rFloat*data){
//			cout<<data[0]<<endl;
//			cout<<"SLP Calc de: "<<num.weight<<endl;
			_rFloat s=0;
			_rInt i;
			for(i=0; i<num.weight; i++)s+=data[i]*weight[i];
			return s+=weight[i];
		}
		public:void train(_rFloat*data,_rFloat rate, _rFloat(*dcostFunction)(_rFloat,_rFloat)/*output-y, target-y*/){
			_rFloat s=0;
			_rInt i;
			_rFloat tmp[num.weight+1];
			for(i=0; i<num.weight; i++)s+=data[i]*weight[i];
			s+=weight[i];

			s=dcostFunction(funcs.activation(s),data[num.weight])*funcs.dactivation(s);
			for(i=0; i<num.weight; i++)tmp[i]=weight[i]-rate*s*data[i];
			tmp[i]=weight[i]-rate*s;

			for(i=num.weight;i+1;i--)weight[i]=tmp[i];
		}

		public:void _p(){ //Temporary Function
			std::cout<<"print > "<<weight<<' '<<num.weight<<'\t';
			for(_rInt i=0;i<=num.weight;i++)std::cout<<weight[i]<<' ';
			std::cout<<endl;
		}
		public:void _s(_rInt idx, _rFloat n){weight[idx]=n;} //Temporary Function
	};
}

namespace MLPa{
	using namespace std;
	template<class _rInt, class _rFloat>
	class MLPa{
		private:struct{
			_rInt layer;
			_rInt maxSlp;
		}num;

		private:typedef struct _sS1{
			SLPa::SLPa<_rInt,_rFloat> slp;
			struct{
				_rFloat c, a, f; //cost, activationFunc, sumFunc
			} cache;
			struct{		//functions for calculatin & training
				_rFloat (*Activation)(_rFloat);
			}funcs;
		}_sS1;

		private:typedef struct _sS0{
			_sS1*slp;
			struct{_rInt slp;} num;
		}_sS0;
		private:_sS0*layer;

		public:MLPa(_rInt numOfLayers, _rInt*numOfPerceptrons, _rInt*numOfInputs){
			this->num.maxSlp=-1;
			//Init 1
			this->num.layer=numOfLayers;
			this->layer=new _sS0[numOfLayers];
			//Init 2
			_rInt i=0,j;
			this->layer[i].num.slp=numOfPerceptrons[i];
			if(this->layer[i].num.slp > this->num.maxSlp)this->num.maxSlp=this->layer[i].num.slp;
			this->layer[i].slp=new _sS1[this->layer[i].num.slp];
			for(j=0;j<this->layer[i].num.slp;j++){
				this->layer[i].slp[j].slp=SLPa::SLPa<_rInt, _rFloat>(numOfInputs[j]);
			}
			
			for(i=1;i<numOfLayers;i++){
				this->layer[i].num.slp=numOfPerceptrons[i];
				this->layer[i].slp=new _sS1[this->layer[i].num.slp];
				if(this->layer[i].num.slp > this->num.maxSlp)this->num.maxSlp=this->layer[i].num.slp;
				for(j=0;j<this->layer[i].num.slp;j++){
					this->layer[i].slp[j].slp=SLPa::SLPa<_rInt, _rFloat>(numOfPerceptrons[i-1]);
				}
			}
			std::cout<<"maxSlp "<<this->num.maxSlp<<std::endl;
		}

		//Member functions
		public:_rFloat*calc(_rFloat*data[], _rFloat*ret){
			std::cout<<"C0"<<std::endl;
			_rFloat tmp[2][num.maxSlp];
			_rInt i=0,j;
			for(j=0;j<layer[i].num.slp;j++)tmp[i&1][j]=layer[i].slp[j].slp.calc(data[j]);
			for(i=1;i<num.layer;i++)for(j=0;j<layer[i].num.slp;j++)tmp[i&1][j]=layer[i].slp[j].slp.calc(tmp[i&1^1]);
			for(j=0;j<layer[num.layer-1].num.slp;j++)ret[j]=tmp[i&1^1][j];
			std::cout<<layer[num.layer-1].num.slp<<std::endl;
			return ret;
		}
		public:void train(_rFloat*data[], _rFloat rate, _rFloat(*dcostFunction)(_rFloat,_rFloat)/*output-y, target-y*/){
			_rInt i,j,k;
			_rFloat tmp[2][num.maxSlp];
			//Forward Propagation
			i=0;
			for(j=0;j<layer[i].num.slp;j++)
				layer[i].slp[j].cache.a=tmp[i&1][j]=layer[i].slp[j].slp.funcs.activation(layer[i].slp[j].cache.f=layer[i].slp[j].slp.calcDeAct(data[j]));
			for(i=1;i<num.layer;i++)for(j=0;j<layer[i].num.slp;j++)
				layer[i].slp[j].cache.a=tmp[i&1][j]=layer[i].slp[j].slp.funcs.activation(layer[i].slp[j].cache.f=layer[i].slp[j].slp.calcDeAct(tmp[i&1^1]));

			//Backward Propagation
			i=num.layer-1;
			
			for(j=0;j<layer[i].num.slp;j++){
				layer[i].slp[j].cache.c = dcostFunction(layer[i].slp[j].cache.a, data[layer[0].num.slp][j]) * layer[i].slp[j].slp.funcs.dactivation(layer[i].slp[j].cache.f), (cout<<"cache.c : "<<layer[i].slp[j].cache.c<<endl);
//				std::cout<<"E "<<dcostFunction(layer[i].slp[j].cache.a, data[layer[0].num.slp][j])<<std::endl;
				for(k=0;k<layer[i].slp[j].slp.num.weight;k++)
					layer[i].slp[j].slp.weight[k] -= rate * layer[i].slp[j].cache.c * layer[i-1].slp[k].cache.a, (std::cout<<"cache.a : "<<layer[i-1].slp[k].cache.a<<'\t'<<std::endl);
				layer[i].slp[j].slp.weight[k] -= rate * layer[i].slp[j].cache.c;
			}
			for(i--;i>0;i--){
				for(j=0;j<layer[i].num.slp;j++){
					for(layer[i].slp[j].cache.c=0,k=0;k<layer[i+1].num.slp;k++)layer[i].slp[j].cache.c+=layer[i+1].slp[k].cache.c * layer[i+1].slp[k].slp.weight[j];
					layer[i].slp[j].cache.c*=layer[i].slp[j].slp.funcs.dactivation(layer[i].slp[j].cache.f);
					layer[i].slp[j].cache.c/=layer[i+1].num.slp;
					for(k=0;k<layer[i].slp[j].slp.num.weight;k++)
						layer[i].slp[j].slp.weight[k]-=rate * layer[i].slp[j].cache.c * layer[i-1].slp[k].cache.a, (std::cout<<layer[i].slp[j].cache.c * layer[i-1].slp[k].cache.a<<'\t'<<std::endl);
					layer[i].slp[j].slp.weight[k]-=rate * layer[i].slp[j].cache.c, (std::cout<<layer[i].slp[j].cache.c * layer[i-1].slp[k].cache.a<<'\t'<<std::endl);
				}
			}
			for(j=0;j<layer[i].num.slp;j++){
				for(layer[i].slp[j].cache.c=0,k=0;k<layer[i+1].num.slp;k++)layer[i].slp[j].cache.c+=layer[i+1].slp[k].cache.c * layer[i+1].slp[k].slp.weight[j];
				layer[i].slp[j].cache.c*=layer[i].slp[j].slp.funcs.dactivation(layer[i].slp[j].cache.f);
				layer[i].slp[j].cache.c/=layer[i+1].num.slp;
				for(k=0;k<layer[i].slp[j].slp.num.weight;k++)
					layer[i].slp[j].slp.weight[k]-=rate * layer[i].slp[j].cache.c * data[j][k], (std::cout<<layer[i].slp[j].cache.c * data[j][k]<<'\t'<<std::endl);
				layer[i].slp[j].slp.weight[k]-=rate * layer[i].slp[j].cache.c, (std::cout<<layer[i].slp[j].cache.c * data[j][k]<<'\t'<<std::endl);
			}
			std::cout<<"E\n";
		}
		public:void setLayerActivationFunc(_rInt layer,_rFloat (*activationFunc)(_rFloat)){
			std::cout<<"setA : "<<layer<<" "<<(void*)activationFunc<<std::endl;
			std::cout<<this->layer[layer].num.slp<<std::endl;
			_rInt i;
			for(i=0;i<this->layer[layer].num.slp;i++)this->layer[layer].slp[i].slp.funcs.activation=activationFunc;
		}
		public:void setLayerDActivationFunc(_rInt layer,_rFloat (*dactivationFunc)(_rFloat)){
			std::cout<<"setDA : "<<layer<<" "<<(void*)dactivationFunc<<std::endl;
			std::cout<<this->layer[layer].num.slp<<std::endl;
			_rInt i;
			for(i=0;i<this->layer[layer].num.slp;i++)this->layer[layer].slp[i].slp.funcs.dactivation=dactivationFunc;
		}
		public:void _p(){
			_rInt i,j;
			std::cout<<"MLP Print"<<std::endl;
			for(i=0;i<num.layer;i++){
				for(j=0;j<layer[i].num.slp;j++)layer[i].slp[j].slp._p();
				std::cout<<std::endl;
			}
			std::cout<<"MLP Print E"<<std::endl;
		}
		public:void _s(_rInt layer, _rInt vertex, _rInt node, _rFloat n){this->layer[layer].slp[vertex].slp._s(node,n);}
	};
}
#endif//__MLP_alpha__
