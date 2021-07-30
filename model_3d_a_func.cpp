#include <iostream>
#include "mpi.h"
#include <string.h>
#include <fstream>
#include <numeric>
#include <vector>
#include <fftw3-mpi.h>
#include<complex>
#include <iomanip>
#include <cmath>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include<complex>
using namespace std;

//compile with
//mpic++ -lfftw3_mpi -lfftw3 -lgsl -lgslcblas -lm -o model_3d_a model_3d_a_func.cpp
//run with GSL_RNG_SEED=123 GSL_RNG_TYPE=mrg 
//mpirun -np 8 model_3d_a


//Начальное время
double t_init (double f0){//Тут аргумент f0 не нужен - он не используется
    double t = 0;
    const double PI = acos(-1.0);
    t = 1.0/(sqrt(pow(10.0,-7.0)*3.0*PI));
    return t;
}


//Масштабный фактор
double a(double t, double f0){//Тут аргумент f0 не нужен - он не используется
    double a;
    const double PI = acos(-1.0);
    a = pow(t, 2.0/3.0)*pow(3.0*PI*pow(10.0,-7), 1.0/3.0);
    return a;
}


//Норма \sum f^2
double norm (vector<complex<double>  > &Psi, int N, double delta, int world_rank, int world_size){
    double sum = 0;
    for (int j = 0; j<Psi.size(); j++){
        sum += pow(abs(Psi.at(j)), 2.0);
    }
    MPI_Allreduce(&sum, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    sum /= delta*delta*delta;
    return sum;
}
  

//Максимальное значение на сетке
double maxPsi (vector<complex<double>  > &Psi){
    double max = abs(Psi.at(0)) ;
    for (int j = 0; j<Psi.size();j++){
        if (abs(Psi.at(j))>max)
            max = abs(Psi.at(j));
    }
    MPI_Allreduce(&max, &max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return max;
}

//Минимальное значение на сетке
double minPsi (vector<complex<double>  > &Psi){
    double min = abs(Psi.at(0)) ;
    for (int j = 0; j<Psi.size();j++){
        if (abs(Psi.at(j))<min)
            min = abs(Psi.at(j));
    }
    MPI_Allreduce(&min, &min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    return min;
}



//Кинетический шаг
void kinetic_step (vector<complex<double>  > &func, double t, double dt, double f0, fftw_plan f, fftw_plan b, int N, double delta, int local_n0, int world_rank){
    const double PI = acos(-1.0);
    complex<double> i (0.0,1.0);
    fftw_execute(f);
    for  (int j = 0; j<(local_n0); j++){
        for (int k = 0; k< N; k++){
        for (int g = 0; g<(N); g++){
        int global_j = j+world_rank*local_n0;
        double k1, k2, k3;
            global_j < N/2 ? k1 = 2.0*PI*(double)(global_j)/(double)(N)/delta : k1 = 2.0*PI*(double)(global_j-N)/(double)(N)/delta;
            k < N/2 ? k2  = 2.0*PI*(double)(k)/(double)(N)/delta : k2 = 2.0*PI*(double)(k-N)/(double)(N)/delta;
            g < N/2 ? k3  = 2.0*PI*(double)(g)/(double)(N)/delta : k3 = 2.0*PI*(double)(g-N)/(double)(N)/delta;
            func.at(j*N*N+k*N+g) *= exp((-i*dt*(pow(k1,2.0)+pow(k2,2.0)+pow(k3,2.0)))/(2.0*pow(a(t,f0),2.0)));
        }
        }}
    fftw_execute(b);
    for  (int j = 0; j<(local_n0); j++){
        for (int k = 0; k< N; k++){
            for (int g = 0; g<(N); g++){
                func.at(j*N*N+k*N+g)/= (double)(N*N*N);
    }}}
}


//Потенциальный шаг
void potential_step (vector<complex<double>  > &func, vector<complex<double>  > &U, double t, double dt, int N, double delta, int local_n0, int world_rank){
    complex<double> i (0.0,1.0);
    for(int j = 0; j<func.size(); j++){
        func.at(j) *= exp (-i*dt*U.at(j));
        }
}


//Получение потенциала
void get_potential (vector<complex<double>  > &func, vector<complex<double>  > &U,  double t, double f0, fftw_plan forward, fftw_plan backward, int N, double delta, int local_n0, int world_rank){
     complex<double> i (0.0,1.0);
    const double PI = acos(-1.0);
    for(int j =0; j<func.size(); j++){
        U.at(j) = 4.0*PI*pow(10,-7)*(pow(abs(func.at(j)),2.0) - f0)/a(t,f0);
    }
  
    fftw_execute(forward);
    
    for  (int j = 0; j<local_n0; j++){
        for(int m = 0; m<N; m++){
            for (int g = 0; g<(N); g++){
                int global_j = j+world_rank*local_n0;
                double k1, k2, k3;
                global_j < N/2 ? k1 = 2.0*PI*(double)(global_j)/(double)(N)/delta : k1 = 2.0*PI*(double)(global_j-N)/(double)(N)/delta;
                m < N/2 ?  k2  = 2.0*PI*(double)(m)/(double)(N)/delta : k2 = 2.0*PI*(double)(m-N)/(double)(N)/delta;
                g < N/2 ?  k3  = 2.0*PI*(double)(g)/(double)(N)/delta : k3 = 2.0*PI*(double)(g-N)/(double)(N)/delta;
                if  ((global_j!=0)||(g!=0)||(m!= 0)){
                    U.at(j*N*N+m*N+g) /= -((k3*k3+k2*k2+k1*k1));}
                else{
                    U.at(j*N*N+m*N+g) = 0.0;}
    }}}
    
    fftw_execute(backward);
    
    for  (int j = 0; j<U.size(); j++){
        U.at(j)/=((double)N*N*N);
    }
}



//Шаг схемы второго порядка
void step_two(vector<complex<double>  > &func, vector<complex<double>  > &U,  double t, double norma, fftw_plan forward, fftw_plan backward, fftw_plan forward1, fftw_plan backward1, int N, double delta, double dt, int local_n0, int world_rank){
    get_potential (func, U,  t, norma , forward, backward, N, delta, local_n0, world_rank);
    potential_step (func, U, t,  dt/2.0, N,delta, local_n0, world_rank);
    t+=dt/2.0;
    kinetic_step (func, t, dt, norma, forward1, backward1, N, delta,local_n0, world_rank);
    t+=dt/2.0;
    get_potential (func, U,  t, norma , forward, backward, N, delta, local_n0, world_rank);
    potential_step (func, U, t,  dt/2.0, N,delta, local_n0, world_rank);
}



//Шаг схемы 4-го порядка
void step_four(vector<complex<double>  > &func, vector<complex<double>  > &U,  double t, double norma, fftw_plan forward, fftw_plan backward, fftw_plan forward1, fftw_plan backward1, int N, double delta, double dt, int local_n0, int world_rank){
    double d[4];
    double c[4];
    d[3] = 0.134496199277431;
    d[2] = -0.224819803079420;
    d[1] = 0.756320000515668;
    d[0] = 0.334003603286321;
    c[3] = 0.515352837431122;
    c[2] = -0.085782019412973;
    c[1] = 0.441583023616466;
    c[0] = 0.128846158365384;
    double p_t = t;
    double k_t = t;
    
    for(int n = 0; n < 4; n++) {
             kinetic_step (func, p_t, c[n]*dt, norma,forward1, backward1, N, delta,local_n0, world_rank);
             k_t += c[n] * dt;
             get_potential (func, U,  k_t, norma , forward, backward, N, delta, local_n0, world_rank);
             potential_step (func, U, k_t,  d[n]*dt, N,delta, local_n0, world_rank);
             p_t += d[n]*dt;
    }
}



//Инициализация физическими параметрами
void init_phys (vector<complex<double>  > &func, double L, double t, int N, double delta, int local_n0, int world_rank, int world_size){
    const double PI = acos(-1.0);
    const gsl_rng_type * T;
    gsl_rng * r;
    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc (T);
    double Delta_R = 2.46*pow(10, -9);
    fftw_plan psi_p;
    psi_p = fftw_mpi_plan_dft_3d(N, N, N, (fftw_complex*) &func.at(0), (fftw_complex*) &func.at(0), MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_MEASURE);
    
  
    
    for (int j = 0; j<local_n0; j++){
    for(int m = 0; m< N; m++){
    for (int g = 0; g< N; g++){
        int global_j = j+world_rank*local_n0;
        double k1, k2, k3;
        global_j < N/2 ? k1 = 2.0*PI*(double)(global_j)/(double)(N)/delta : k1 = 2.0*PI*(double)(global_j-N)/(double)(N)/delta;
        m < N/2 ?  k2  = 2.0*PI*(double)(m)/(double)(N)/delta : k2 = 2.0*PI*(double)(m-N)/(double)(N)/delta;
        g < N/2 ?  k3  = 2.0*PI*(double)(g)/(double)(N)/delta : k3 = 2.0*PI*(double)(g-N)/(double)(N)/delta;
        double k_2 = pow(k1,2.0) + pow(k2,2.0)+ pow(k3, 2.0);
        double k_3 = pow(pow(k1,2.0) + pow(k2,2.0)+ pow(k3, 2.0), 3.0/2.0);
        double kmax = 2.0*PI*(double)(32)/(double)(N)/delta;
        if(((k1==0)&&(k2==0)&&(k3==0))||((abs(3.0*sqrt(k_2)*t/2.0)<10.0)||abs(3.0*sqrt(k_2)*t/2.0)>100.0)){
            func.at(j*N*N+m*N+g) = 0.0;
        }
        else{
        double delta_i = Delta_R/(4.0*PI)/k_3*pow(L,3.0)/(pow(2.0*PI, 3.0));
        double sigma = sqrt(pow(-2.0/3.0 * (4.0 + 9.0*k_2*pow(t,2.0)/4.0),2.0) * delta_i);
        func.at(j*N*N+m*N+g) =  gsl_ran_gaussian(r, sigma)/2.0;
        }
    
    }}}
    
    double f_k = norm (func, N,  delta, world_rank,  world_size);
    
    fftw_execute(psi_p);
      
    
    
    for(int j = 0; j<func.size(); j++){
        func.at(j)=1.0+(func.at(j)*pow(2.0*PI/L,3.0));
}
}


//Вывод данных в файл. Для каждого процессора и для каждого шага создается свой файл
void data(vector<complex<double>  > &func, int N, double delta, int local_n0, int world_rank, int step, double t){
    ofstream data;
    string name = "/Users/glebsuzdalov/Desktop/МГУ/2 курс/2 семестр/курсовая/test_parallel/data/data" + to_string(world_rank)+ "_" + to_string(step) + ".txt";
    data.open(name);
    //cout<<"open"<<" "<<world_rank<<endl;
    for (int j = 0; j<local_n0; j++){
    for(int m = 0; m< N; m++){
    for (int g = 0; g< N; g++){
        int global_j = j+world_rank*local_n0;
        if((j%2==0)&&(m%2==0)&&(g%2==0)){
            data<<fixed << setprecision(16)<<global_j*delta*a(t, 1.0)<<" "<<m*delta*a(t, 1.0)<<" "<<g*delta*a(t, 1.0)<<" "<<abs(func.at(j*N*N+m*N+g))<<endl;}
        
    }}}
    data.close();
}


//Проверка - n шагов, затем делается n/2 с шагом 2dt и так далее
void check(vector<complex<double>  > &rem,  double t, double norma, int N, double delta, double dt, int local_n0, int world_rank){
    double error;
    fftw_plan forward, backward, forward1, backward1, forward2, backward2;
    vector<complex<double>  > func1 (N*N*local_n0);
    vector<complex<double>  > func2 (N*N*local_n0);
    vector<complex<double>  > U (N*N*local_n0);
    
    forward = fftw_mpi_plan_dft_3d(N, N, N, (fftw_complex*) &U.at(0), (fftw_complex*) &U.at(0), MPI_COMM_WORLD, FFTW_FORWARD, FFTW_MEASURE);
    backward = fftw_mpi_plan_dft_3d(N, N, N, (fftw_complex*) &U.at(0), (fftw_complex*) &U.at(0), MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_MEASURE);
    forward1 = fftw_mpi_plan_dft_3d(N, N, N, (fftw_complex*) &func1.at(0), (fftw_complex*) &func1.at(0), MPI_COMM_WORLD, FFTW_FORWARD, FFTW_MEASURE);
    backward1 = fftw_mpi_plan_dft_3d(N, N, N, (fftw_complex*) &func1.at(0), (fftw_complex*) &func1.at(0), MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_MEASURE);
    forward2 = fftw_mpi_plan_dft_3d(N, N, N, (fftw_complex*) &func2.at(0), (fftw_complex*) &func2.at(0), MPI_COMM_WORLD, FFTW_FORWARD, FFTW_MEASURE);
    backward2 = fftw_mpi_plan_dft_3d(N, N, N, (fftw_complex*) &func2.at(0), (fftw_complex*) &func2.at(0), MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_MEASURE);
    
    double g_t = t;
    double p_t = t;
    double k_t = t;
        
    for (int j = 0; j<func1.size(); j++) {
         func1.at(j) = rem.at(j);
     }
     
     for(int j = 0; j < 64; j++){
         step_four(func1, U,  t, norma, forward, backward, forward1, backward1, N, delta, dt, local_n0, world_rank);
         t+=dt;
     }
    
     
     for(double k = 1.0 ; k<65.0; k*=2.0){
     double t2 = g_t;
     double dt2 = dt*k;
     double p_t2 = t2;
     double k_t2 = t2;
     int tr = 64/((int)k);
     for (int j = 0; j<rem.size(); j++) {
             func2.at(j) = rem.at(j);
     }
     
     for(int j = 0; j<tr; j++){
         step_four(func2, U,  t2, norma, forward, backward, forward2, backward2, N, delta, dt2, local_n0, world_rank);
         t2+=dt2;
     }
         
     double maxdiff = 0;
     for(int j = 0; j<func1.size(); j++){
         if( abs(func2.at(j)-func1.at(j))>maxdiff){
             maxdiff =  abs(func2.at(j)-func1.at(j));
         }
     }
     MPI_Allreduce(&maxdiff, &maxdiff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
     if(world_rank == 0){
     cout<<fixed << setprecision(16)<<dt2<<" "<<maxdiff<<endl;
     }
    
}}


//Сборка данных с каждого процессора на 1 (сейчас она делает не совсем это, но эту функцию можно запустить другой программой и объединять файлы после вычислений) 
void finish_data(int world_rank, int steps, int world_size){
    if(world_rank == 0){
        for (int j = 0; j<steps; j++){
            ofstream out("/Users/glebsuzdalov/Desktop/МГУ/2 курс/2 семестр/курсовая/test_parallel/data/fulldata" + to_string(j) + ".txt");
            for(int m = 0; m<world_size; m++){
                string str;
                string name = "/Users/glebsuzdalov/Desktop/МГУ/2 курс/2 семестр/курсовая/test_parallel/data/data" + to_string(m)+ "_" + to_string(j) + ".txt";
                cout<<name<<endl;
                ifstream in(name, ios::in);
                int b = 0;
                while (in >> str){
                    if(b%4 == 0){
                        out<<endl;
                    }
                    out << str<<" ";
                    b++;
                }
                in.close();
        }
            out.close();
    }
    }
}



int main(int argn, char **argv)
{
    
    //Инициализация окружения MPI
    MPI_Init(&argn, &argv);
    fftw_mpi_init();
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    //Переменные

    const double size = 100.0;
    const ptrdiff_t N = 64;
    fftw_plan forward, backward;
    ptrdiff_t alloc_local, local_n0, local_0_start;
    const double PI = acos(-1.0);
    complex<double> i (0.0,1.0);
    local_n0 = N/((ptrdiff_t)world_size);
    local_0_start = local_n0*(ptrdiff_t)world_size;
    double delta = size/(double)N;
    fftw_plan forward1, backward1, forward2, backward2;
    double t;
    double dt = 0.01;
    
    //FFTW
    //Выделение массива
    vector<complex<double>  > func1 (N*N*local_n0);
    vector<complex<double>  > func2 (N*N*local_n0);
    vector<complex<double>  > rem (N*N*local_n0);
    vector<complex<double>  > U (N*N*local_n0);
    vector<complex<double>  > Ro (N*N*local_n0);
    vector<complex<double>  > remU (N*N*local_n0);
    vector<double  > absfunc (N*N*local_n0);
    
    //Планы
    forward = fftw_mpi_plan_dft_3d(N, N, N, (fftw_complex*) &U.at(0), (fftw_complex*) &U.at(0), MPI_COMM_WORLD, FFTW_FORWARD, FFTW_MEASURE);
    backward = fftw_mpi_plan_dft_3d(N, N, N, (fftw_complex*) &U.at(0), (fftw_complex*) &U.at(0), MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_MEASURE);
    forward1 = fftw_mpi_plan_dft_3d(N, N, N, (fftw_complex*) &func1.at(0), (fftw_complex*) &func1.at(0), MPI_COMM_WORLD, FFTW_FORWARD, FFTW_MEASURE);
    backward1 = fftw_mpi_plan_dft_3d(N, N, N, (fftw_complex*) &func1.at(0), (fftw_complex*) &func1.at(0), MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_MEASURE);
    forward2 = fftw_mpi_plan_dft_3d(N, N, N, (fftw_complex*) &func2.at(0), (fftw_complex*) &func2.at(0), MPI_COMM_WORLD, FFTW_FORWARD, FFTW_MEASURE);
    backward2 = fftw_mpi_plan_dft_3d(N, N, N, (fftw_complex*) &func2.at(0), (fftw_complex*) &func2.at(0), MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_MEASURE);
    
    
    //Файлы
    
    ofstream error1;
    if(world_rank == 0){
    error1.open("/Users/glebsuzdalov/Desktop/МГУ/2 курс/2 семестр/курсовая/test_parallel/Shredinger parallel/error.txt");
    }
    
    
    //Инициализация
    t = t_init(1.0);
    int step = 0;
    init_phys (func1, size, t, N, delta, (int)local_n0, (int)world_rank, (int)world_size);
    data(func1,  N,  delta, local_n0, world_rank, step, t);
    
    
    double norma = norm(func1, (int)local_n0, size, world_rank, world_size);
    norma *=  delta*delta*delta;
    t = t_init(norma);
    cout<<"F_0 is "<<norma<<endl;
     
    
    double  f0 = norm(func1, (int)local_n0, size, world_rank, world_size)*delta*delta*delta;
    if(world_rank == 0){
        cout<<fixed << setprecision(16)<<"norm "<<f0<<endl;
    }
    
    
    double d[4];
    double c[4];
    
    d[3] = 0.134496199277431;
    d[2] = -0.224819803079420;
    d[1] = 0.756320000515668;
    d[0] = 0.334003603286321;
    c[3] = 0.515352837431122;
    c[2] = -0.085782019412973;
    c[1] = 0.441583023616466;
    c[0] = 0.128846158365384;
    
    double worktime;
    
    worktime = MPI_Wtime();
   
    //Цикл по времени
    dt = 10.0;
    for (int h = 0; h<100000; h++){
        double maxP, minP;
        maxP = maxPsi(func1);
        minP = minPsi(func1);
                
       step_four(func1, U,  t, norma, forward, backward, forward1, backward1, N, delta, dt, local_n0, world_rank);
       t+=dt;
        
        if(h%1000 == 0){
            data(func1,  N,  delta, local_n0, world_rank, step, t);
            step +=1;
        }
        
        if(world_rank == 0){
            cout<<"Time is "<<t<<endl;
            cout<<"Max "<<maxP<<endl;
            cout<<"Min "<<minP<<endl;
            
        }
        }
    worktime = MPI_Wtime() - worktime;
    if(world_rank == 0){
        cout<<"Program time is "<<worktime<<endl;
    }
    finish_data(world_rank, step+1, world_size);
    fftw_mpi_cleanup();
    MPI_Finalize();
    }

