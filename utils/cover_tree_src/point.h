#include<stdio.h>
#include<stdlib.h>
#include <vector>

typedef std::vector<double> point;
typedef double (*distfn)(const point&, const point&, double, const double*);

double complete_distance(const point &v1, const point &v2);
double distance_bounded(const point &v1, const point &v2, double upper_bound, const double *NO_PARAMS);
std::vector<point> parse_points(FILE *input);
void print(const point &p);
