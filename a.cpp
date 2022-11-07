#include<iostream>
#include"MLP_alpha/MLP_alpha.h"
using namespace std;
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
		cout<<a.calc(l0)<<endl;
	}
	cout<<"H1"<<endl;
	{
		int l0[]={3,4,5,2}, l1[]={2,3,4};
		MLPa::MLPa<int, long double> a(4, l0, l1);
	}
}
