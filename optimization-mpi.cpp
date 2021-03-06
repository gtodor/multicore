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
#include <omp.h>
#include "interval.h"
#include "functions.h"
#include "minimizer.h"

using namespace std;

//This structure contains the x and y domains where we will
//evaluate the function
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
  // Stores the name of the function as a c string and is used to send
  // the function name with mpi which doesn't recognize c++ strings   
  char choice[100];
  // The information on the function chosen (pointer and initial box)
  opt_fun_t fun;
  
  bool good_choice;

  //variable storing the number of subdomains of the initial box
  int subdomains;

  //the number of processors in the communicator and the rank of
  //the current machine
  int numprocs, rank;

  //Initializing MPI
  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  //variable that holds the length of the function name
  //needed because a client needs to know the length of the
  //c string to read 
  int len = 0;

  //variable storing the number of boxes each client has to process
  //besides the one it is given by default
  int extra = 0;

  //initializing the choice array
  memset(choice,0,100);

  //variable that stores the local min_ub variable of each client
  //it is used in MPI_Allreduce to define the new general min_ub
  //and then it is replaced by the new min_ub 
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

    //Ask for the minimum number of subdomains we want to split the initial box
    cout<<"Number of subdomains? ";
    cin>>subdomains;

    //Transform function name in a c type string to use in broadcast
    strcpy(choice,choice_fun.c_str());
    len = strlen(choice)+1;
  }

  //send the length of the function name to all clients in the communicator
  MPI_Bcast(&len,1,MPI_INT,0,MPI_COMM_WORLD);
  
  //send the precision to everyone
  MPI_Bcast(&precision,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
  
  //send the function name to everyone
  MPI_Bcast(choice,len,MPI_CHAR,0,MPI_COMM_WORLD);

  //send the number of subdomains to everyone
  MPI_Bcast(&subdomains,1,MPI_INT,0,MPI_COMM_WORLD);

  //if the machine's rank is not 0 it has to load the function based by the
  //function name it received from rank 0 by broadcast
  if(rank != 0){
    choice_fun = choice;
    fun = functions.at(choice_fun);
  }

  
  //each machine is dividing the initial box in "subdomains" number of boxes
  //and it is going to use only some of them
  
  box initbox = {fun.x,fun.y};
  vector<box>* boxes = new vector<box>(1,initbox);

  //helper vector : at each iteration of the first while loop we split in four
  //every box and store them in the temp vector, then we swap the two vectors
  vector<box>* temp = new vector<box>();
  
  while(boxes->size() < subdomains){
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

  //we compute the number of boxes every client has to process
  //if some boxes remain we process them afterwards
  if(boxes->size() > numprocs){
    int remaining = boxes->size() - numprocs;
    int quotient = remaining / numprocs;
    extra = quotient;
    MPI_Bcast(&extra,1,MPI_INT,0,MPI_COMM_WORLD);
  }

  //the total number of boxes to process
  int toanalyse = 1 + extra;
  
  
  omp_set_num_threads(toanalyse);

  //the indexes of the boxes each client has to process are computed based on the rank
#pragma omp parallel for 
  for(int i = rank * toanalyse; i<rank* toanalyse + toanalyse; i++){

    //minimizer list private to the thread
    //if not used there are some incorect accesses to the minimums variable causing
    //segmentation fault errors we didn't manage to solve
    minimizer_list m;
    
    minimize(fun.f,boxes->at(i).x,boxes->at(i).y,precision,localmin,m);  

    //After processing the box every processor share it's localmin to everyone
    //we reduce all these values and broadcast to every client using MPI_Allreduce
    //after this function the variable min_ub will contain the new minimum
    MPI_Allreduce(&localmin,&min_ub,1,MPI_DOUBLE,MPI_MIN,MPI_COMM_WORLD);
    
  

    //because there is a new minimum we need to clear some values from the minimize list
    auto discard_begin = m.lower_bound(minimizer{0,0,min_ub,0});   
    m.erase(discard_begin,m.end());
    
  
    localmin = min_ub;

    //insert all elements from local minimizer list to minimums
    for(auto& min : m){
      #pragma omp critical
      minimums.insert(min);
    }
    
  }

  //check for values not respecting the last min_ub computed in the previous loop
  auto discard_begin = minimums.lower_bound(minimizer{0,0,min_ub,0});
  minimums.erase(discard_begin,minimums.end());
  
    

  int max_rank = numprocs ;
  int rest = boxes->size() - numprocs*toanalyse;
  
  if(rest > 0){
    
    //each remaining box is processed by a particular processor
    if(max_rank * toanalyse + rank < boxes->size()){
      
      minimize(fun.f,boxes->at(max_rank * toanalyse + rank).x,boxes->at(max_rank * toanalyse + rank).y,precision,localmin,minimums);
      
      MPI_Allreduce(&localmin,&min_ub,1,MPI_DOUBLE,MPI_MIN,MPI_COMM_WORLD);
      
      auto discard_begin = minimums.lower_bound(minimizer{0,0,min_ub,0});
      minimums.erase(discard_begin,minimums.end());
	
      localmin = min_ub;
    }
  }
  
  // Displaying all potential minimizers
  copy(minimums.begin(),minimums.end(),
       ostream_iterator<minimizer>(cout,"\n"));    
  cout << "Number of minimizers: " << minimums.size() <<" at rank "<<rank<<endl;
  cout << "Upper bound for minimum: " << min_ub <<" at rank "<<rank<<endl;

  MPI_Finalize();
}
