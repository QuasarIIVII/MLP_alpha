//#pragma once
#ifndef __MLP_alpha__
#define __MLP_alpha__
#include<iostream>
#include<iomanip>
#include<math.h>
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
			_rFloat (*activatoin)(_rFloat);
			_rFloat (*dactivatoin)(_rFloat);
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
			this->funcs.activatoin=&tanhl; // ========Need to handle NullPtrException========
			this->funcs.dactivatoin=&_dtanh;
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
			_rFloat s=0;
			_rInt i;
			for(i=0; i<num.weight; i++)s+=data[i]*weight[i];
			s+=weight[i];
			return funcs.activatoin(s);
		}
		public:void train(_rFloat*data,_rFloat rate, _rFloat(*dcostFunction)(_rFloat,_rFloat)/*output-y, target-y*/){
			_rFloat s=0;
			_rInt i;
			_rFloat tmp[num.weight+1];
			for(i=0; i<num.weight; i++)s+=data[i]*weight[i];
			s+=weight[i];

			s=dcostFunction(funcs.activatoin(s),data[num.weight])*funcs.dactivatoin(s);
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
	template<class _rInt, class _rFloat>
	class MLPa{
		private:struct{
			_rInt layer;
		}num;

		private:typedef struct _sS1{
			SLPa::SLPa<_rInt,_rFloat> slp;
			struct{		//functions for calculatin & training
				_rFloat (*Activation)(_rFloat);
			}funcs;
		}_sS1;

		private:typedef struct _sS0{
			_sS1*slp;
			struct{_rInt slp;} num;
		}_sS0;
		private:_sS0*layer;

		_rFloat (*dCost)(_rFloat,_rFloat);//output-y, target-y

		public:MLPa(_rInt numOfLayers, _rInt*numOfPerceptrons, _rInt*numOfInputs){
			//Init 1
			this->num.layer=numOfLayers;
			this->layer=new _sS0[numOfLayers];
			//Init 2
			_rInt i=0,j;
			this->layer[i].num.slp=numOfPerceptrons[i];
			this->layer[i].slp=new _sS1[this->layer[i].num.slp];
			for(j=0;j<this->layer[i].num.slp;j++)this->layer[i].slp[j].slp=SLPa::SLPa<_rInt, _rFloat>(numOfInputs[j]);
			
			for(i=1;i<numOfLayers;i++){
				this->layer[i].num.slp=numOfPerceptrons[i];
				this->layer[i].slp=new _sS1[this->layer[i].num.slp];
				for(j=0;j<this->layer[i].num.slp;j++)this->layer[i].slp[j].slp=SLPa::SLPa<_rInt, _rFloat>(numOfPerceptrons[i-1]);
			}
		}
	};
}
#endif//__MLP_alpha__
