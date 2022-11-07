#include<iostream>
#include"MLP_alpha/MLP_alpha.h"
using namespace std;
long double dCost(long double o, long double t){return 2*(o-t);}
int main(){
	{
		cout<<"H"<<endl;
		SLPa::SLPa<int, long double> a;
		a._p();
		a=SLPa::SLPa<int, long double>(12);
		a._p();
		a._s(3,0.7);
		SLPa::SLPa<int, long double> b=a;
		b._p();
	}
	{
		SLPa::SLPa<int, long double> a(2);
		a._s(0,1.0);
		a._s(1,1.0);
		a._p();
		long double l0[]={2,-2};
		long double l1[4][3]={{0,0,-1},{0,1,-1},{1,0,-1},{1,1,1}};
		for(int64_t i=1<<20;i;i--)for(int j=3;j+1;j--)a.train(l1[j],0.01,dCost);
		cout<<a.calc(l1[0])<<' '<<a.calc(l1[1])<<' '<<a.calc(l1[2])<<' '<<a.calc(l1[3])<<endl;
	}
	cout<<"H1"<<endl;
	{
		int l0[]={3,4,5,2}, l1[]={2,3,4};
		MLPa::MLPa<int, long double> a(4, l0, l1);
	}
}
