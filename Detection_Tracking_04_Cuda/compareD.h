#ifndef COMPARED_H
#define COMPARED_H

#define EPSS 1e-4
static int compareD( double x, double y ) { return ( x <= y + EPSS ) ? ( x + EPSS < y ) ? -1 : 0 : 1; }

#endif // COMPARED_H
