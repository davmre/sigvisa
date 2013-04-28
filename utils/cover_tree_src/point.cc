#include "point.h"
#define NDEBUG
#include<assert.h>
#include<math.h>
#include <string.h>
#include <iostream>
using namespace std;


float distance_bounded(const point &p1, const point &p2, float upper_bound)
{
  float sum = 0.;
  for(point::size_type i = 0; i != p1.size(); i+=1) {
    float d1 = p1[i] - p2[i];
    sum += d1 * d1;
  }
  return sqrt(sum);
}

vector<point > parse_points(FILE *input)
{

  char c;

  // read the number of points in the file
  int npoints=0;
  while ( (c = getc(input)) != EOF )
    if (c=='\n')
      npoints++;
  rewind(input);
  printf("read file, found %d points\n", npoints);
  // read the length of an arbitrary point
  int point_len=0;
  while ((c = getc(input)) != '\n' ) {
    while (c != '0' && c != '1' && c != '2' && c != '3'
	   && c != '4' && c != '5' && c != '6' && c != '7'
	   && c != '8' && c != '9' && c != '\n' && c != EOF && c != '-')
      c = getc(input);
    if (c != '\n' && c != EOF) {
      ungetc(c,input);
      float f;
      fscanf(input, "%f",&f);
      point_len++;
    } else if (c == '\n') {
      break;
    }
  }
  rewind(input);
  printf("read file, found point len %d\n" , point_len);

  // now read all points into memory
  vector<point> parsed(npoints, point(point_len));
  int i=0, j=0;
  while ( (c = getc(input)) != EOF ) {
    ungetc(c,input);

    while ((c = getc(input)) != '\n' ) {
      while (c != '0' && c != '1' && c != '2' && c != '3'
	     && c != '4' && c != '5' && c != '6' && c != '7'
	     && c != '8' && c != '9' && c != '\n' && c != EOF && c != '-')
	c = getc(input);
      if (c != '\n' && c != EOF) {
	ungetc(c,input);
	float f;
	fscanf(input, "%f",&f);
	parsed[i][j++] = f;
      }
    }

    i++;

    if (j != point_len) {
      printf("Can't handle vectors of differing length %d vs %d, bailing\n", j, point_len);
      exit(0);
    }
    j=0;
  }
  return parsed;
}

void print(const point &p)
{
  for (point::size_type i = 0; i<p.size(); i++)
    printf("%f ",p[i]);
  printf("\n");
}
