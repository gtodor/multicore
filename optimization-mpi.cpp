/*
  Branch and bound algorithm to find the minimum of continuous binary 
  functions using interval arithmetic.

  Sequential version

  Author: Frederic Goualard <Frederic.Goualard@univ-nantes.fr>
  v. 1.0, 2013-02-15
*/

#include <iostream>
#include <iterator>
#include <string>
#include <stdexcept>
#include <vector>
#include <mpi.h>
#include "interval.h"
#include "functions.h"
#include "minimizer.h"

using namespace std;

struct box{
  interval x;
  interval y;
};


// Split a 2D box into four subboxes by splitting each dimension
// into two equal subparts
void split_box(const interval& x, const interval& y,
	       interval &xl, interval& xr, interval& yl, interval& yr)
{
  double xm = x.mid();
  double ym = y.mid();
  xl = interval(x.left(),xm);
  xr = interval(xm,x.right());
  yl = interval(y.left(),ym);
  yr = interval(ym,y.right());
}

// Branch-and-bound minimization algorithm
void minimize(itvfun f,  // Function to minimize
	      const interval& x, // Current bounds for 1st dimension
	      const interval& y, // Current bounds for 2nd dimension
	      double threshold,  // Threshold at which we should stop splitting
	      double& min_ub,  // Current minimum upper bound
	      minimizer_list& ml) // List of current minimizers
{
  interval fxy = f(x,y);
  
  if (fxy.left() > min_ub) { // Current box cannot contain minimum?
    return ;
  }

  if (fxy.right() < min_ub) { // Current box contains a new minimum?
    min_ub = fxy.right();
    // Discarding all saved boxes whose minimum lower bound is 
    // greater than the new minimum upper bound
    auto discard_begin = ml.lower_bound(minimizer{0,0,min_ub,0});
    ml.erase(discard_begin,ml.end());
  }

  // Checking whether the input box is small enough to stop searching.
  // We can consider the width of one dimension only since a box
  // is always split equally along both dimensions
  if (x.width() <= threshold) { 
    // We have potentially a new minimizer
    ml.insert(minimizer{x,y,fxy.left(),fxy.right()});
    return ;
  }

  // The box is still large enough => we split it into 4 sub-boxes
  // and recursively explore them
  interval xl, xr, yl, yr;
  split_box(x,y,xl,xr,yl,yr);

  minimize(f,xl,yl,threshold,min_ub,ml);
  minimize(f,xl,yr,threshold,min_ub,ml);
  minimize(f,xr,yl,threshold,min_ub,ml);
  minimize(f,xr,yr,threshold,min_ub,ml);
}


int main(int argc,char** argv)
{
  cout.precision(16);
  // By default, the currently known upper bound for the minimizer is +oo
  double min_ub = numeric_limits<double>::infinity();
  // List of potential minimizers. They may be removed from the list
  // if we later discover that their smallest minimum possible is 
  // greater than the new current upper bound
  minimizer_list minimums;
  // Threshold at which we should stop splitting a box
  double precision;

  // Name of the function to optimize
  string choice_fun;
  char choice[100];
  // The information on the function chosen (pointer and initial box)
  opt_fun_t fun;
  
  bool good_choice;

  int numprocs, rank;
  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  //cout<<"numprocs = "<<numprocs<<endl;
  //cout<<"rank = "<<rank<<endl;
  int len = 0;
  int extra = 0;
  memset(choice,0,100);

  double localmin = min_ub;

  
  // Asking the user for the name of the function to optimize
  if(rank == 0){
    do {
      good_choice = true;
      
      cout << "Which function to optimize?\n";
      cout << "Possible choices: ";
      for (auto fname : functions) {
	cout << fname.first << " ";
      }
      cout << endl;
      cin >> choice_fun;
    
      try {
	fun = functions.at(choice_fun);
      } catch (out_of_range) {
	cerr << "Bad choice" << endl;
	good_choice = false;
      }
    } while(!good_choice);
    
    // Asking for the threshold below which a box is not split further
    cout << "Precision? ";
    cin >> precision;

    //Transform function name in a c type string to use in broadcast
    strcpy(choice,choice_fun.c_str());
    len = strlen(choice)+1;
    cout<<len<<endl;
  }
  
  MPI_Bcast(&len,1,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(&precision,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
  MPI_Bcast(choice,len,MPI_CHAR,0,MPI_COMM_WORLD);

  if(rank != 0){
    choice_fun = choice;
    cout<<"rank "<<rank<<"received precision ="<<precision<<endl;
    fun = functions.at(choice_fun);
  }

  //divide the initial box in n*n domain boxes n = nb of processors
  //each processor is using n domains

  box initbox = {fun.x,fun.y};
  vector<box>* boxes = new vector<box>(1,initbox);
  vector<box>* temp = new vector<box>();
  
  while(boxes->size() < numprocs){
    while(!boxes->empty()){
      interval x = boxes->back().x;
      interval y = boxes->back().y;
      boxes->pop_back();
      interval xl,yl,xr,yr;
      split_box(x,y,xl,xr,yl,yr);
      temp->push_back(box{xl,yl});
      temp->push_back(box{xl,yr});
      temp->push_back(box{xr,yl});
      temp->push_back(box{xr,yr});
    }
    vector<box>* t;
    t = boxes;
    boxes = temp;
    temp = t;
  }

  if(boxes->size() > numprocs){
    int remaining = boxes->size() - numprocs;
    int quotient = remaining / numprocs;
    extra = quotient;
    MPI_Bcast(&extra,1,MPI_INT,0,MPI_COMM_WORLD);
  }

  int toanalyse = 1 + extra;
  cout<<"number of boxes to analyze = "<<toanalyse<<endl; 

  //#pragma omp parallel for
  for(int i = rank * toanalyse; i<rank* toanalyse + toanalyse; i++){
    cout<<"rank "<<rank<<" : step "<<i<<" before minimize"<<endl;
    minimize(fun.f,boxes->at(i).x,boxes->at(i).y,precision,localmin,minimums);
    
    cout<<"rank "<<rank<<" : step "<<i<<" minimize ended"<<endl;

    
    MPI_Allreduce(&localmin,&min_ub,1,MPI_DOUBLE,MPI_MIN,MPI_COMM_WORLD);
    cout<<"rank "<<rank<<" : step "<<i<<" alreduce ended"<<endl;
    if(localmin > min_ub){
      auto discard_begin = minimums.lower_bound(minimizer{0,0,min_ub,0});
      minimums.erase(discard_begin,minimums.end());
    }
    cout<<"rank "<<rank<<" : step "<<i<<" discarding wrong minimum box finished"<<endl;
    localmin = min_ub;
    
  }


  int max_rank = numprocs ;
  int rest = boxes->size() - numprocs*toanalyse;
  cout<<"threr are "<<boxes->size()<<" boxes to analyse"<<endl;
  cout<<"numprocs = "<<numprocs<<endl;
  cout<<"rest = "<<rest<<endl;
  if(rest > 0){
    int r = 0;
    for(int j = 0; j < rest; j++){
      if(rank == r){
	minimize(fun.f,boxes->at(max_rank * toanalyse + j).x,boxes->at(max_rank * toanalyse + j).y,precision,localmin,minimums);
	cout<<endl<<endl<<"rank "<<rank<<" extra box"<<endl;
      }
      
      MPI_Allreduce(&localmin,&min_ub,1,MPI_DOUBLE,MPI_MIN,MPI_COMM_WORLD);
      if(localmin > min_ub){
	auto discard_begin = minimums.lower_bound(minimizer{0,0,min_ub,0});
	minimums.erase(discard_begin,minimums.end());
      }
      localmin = min_ub;
      
      r++;
    }
  }
  /*
  MPI_Allreduce(&localmin,&min_ub,1,MPI_DOUBLE,MPI_MIN,MPI_COMM_WORLD);

  auto discard_begin = minimums.lower_bound(minimizer{0,0,min_ub,0});
  minimums.erase(discard_begin,minimums.end());
  */
  
  // Displaying all potential minimizers
  copy(minimums.begin(),minimums.end(),
       ostream_iterator<minimizer>(cout,"\n"));    
  cout << "Number of minimizers: " << minimums.size() <<" at rank "<<rank<<endl;
  cout << "Upper bound for minimum: " << min_ub <<" at rank "<<rank<<endl;

  MPI_Finalize();
}
