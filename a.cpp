#include<iostream>
#include"MLP_alpha/MLP_alpha.h"
#define _A2P(T) (T*)(T[])
using namespace std;
long double dCost(long double o, long double t){return 2*(o-t);}
long double ELU(long double x){return x<0 ? expl(x)-1 : x;}
long double dELU(long double x){return x<0 ? expl(x) : 1;}
long double _tanh(long double x){return (tanh(x)+1)/2;}
int main(){
	cout<<"Powered by Rúhûlù"<<endl;
//	{
//		cout<<"H"<<endl;
//		SLPa::SLPa<int, long double> a;
//		a._p();
//		a=SLPa::SLPa<int, long double>(12);
//		a._p();
//		a._s(3,0.7);
//		SLPa::SLPa<int, long double> b=a;
//		b._p();
//	}
//	{
//		SLPa::SLPa<int, long double> a(2);
//		a._s(0,1.0);
//		a._s(1,1.0);
//		a._p();
//		long double l0[]={2,-2};
//		long double l1[4][3]={{0,0,-1},{0,1,1},{1,0,1},{1,1,1}};
//		for(int64_t i=1<<10;i;i--)for(int j=3;j+1;j--)a.train(l1[j],0.01,dCost);
//		cout<<a.calc(l1[0])<<' '<<a.calc(l1[1])<<' '<<a.calc(l1[2])<<' '<<a.calc(l1[3])<<endl;
//	}
	cout<<"H1"<<endl;
//	{
//		int l0[]={3,4,5,2}, l1[]={2,3,4,7,3,1,7};
//		long double tl0={};
//		long double*l2[]={
//			_A2P(long double){2.0,4.0,7.7},
//			_A2P(long double){2.0,4.0,7.0,4.2},
//			_A2P(long double){-8.0,2.0,-1.0,4.0,3.2},
//			_A2P(long double){2.0,4.0}};
//		long double*l3[]={};
//
//		MLPa::MLPa<int, long double> a(2, l0, l1);
//		a.setLayerActivationFunc(0,&linear);
//		a.setLayerActivationFunc(1,&linear);
//		a.setLayerActivationFunc(2,&linear);
//		a.setLayerActivationFunc(3,&linear);
//		cout<<"linear : "<<(void*)linear<<endl;
//		long double ret[16];
//		cout<<l2[0]<<endl;
//		a.calc(l2,ret);
//		for(int i=0;i<16;i++)cout<<ret[i]<<' ';cout<<endl;
//	}
	{
		long double tl0[]={0,0}, tl0r[]={0};
		long double tl1[]={0,1}, tl1r[]={0};
		long double tl2[]={1,0}, tl2r[]={0};
		long double tl3[]={1,1}, tl3r[]={0};
		long double *l2[][3]={
			{tl0,tl0,tl0r},
			{tl1,tl1,tl1r},
			{tl2,tl2,tl2r},
			{tl3,tl3,tl3r},
		};
		int l0[]={2,2,1}, l1[]={2, 2};
		long double r[16];

		MLPa::MLPa<int, long double> a(3,l0,l1);
		a.setLayerActivationFunc(0,&ELU);
		a.setLayerActivationFunc(1,&ELU);
		a.setLayerActivationFunc(2,&_tanh);

		a.setLayerDActivationFunc(0,&dELU);
		a.setLayerDActivationFunc(1,&dELU);
		a.setLayerDActivationFunc(2,&_dtanh);

		a._p();

		for(int t=1<<13;t;t--)for(int i=0;i<4;i++)a.train(l2[i], 0.001, dCost);

		a._p();

		for(int i=0;i<4;i++){
			a.calc(l2[i], r);
			for(int j=0;j<16;j++)cout<<r[j]<<' ';cout<<endl;
		}
	}
	cout<<"Powered by Rúhûlù"<<endl;
}
