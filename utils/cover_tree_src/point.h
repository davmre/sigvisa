#include<stdio.h>
#include<stdlib.h>
#include <vector>

typedef std::vector<float> point;
typedef float (*distfn)(const point&, const point&, float);

float complete_distance(const point &v1, const point &v2);
float distance_bounded(const point &v1, const point &v2, float upper_bound);
std::vector<point> parse_points(FILE *input);
void print(const point &p);
