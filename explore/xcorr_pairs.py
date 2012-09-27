import numpy as np
import sys, itertools



def xcorr(a, b, f1, f2):

    a = (a - np.mean(a)) / (np.std(a) * np.sqrt(len(a)))
    b = (b - np.mean(b)) / (np.std(b) * np.sqrt(len(b)))

    xc = np.correlate(a, b, 'full')
    N = len(a)
    unbiased = np.array([float(N)/(N- np.abs(N-i)) for i in range(1, 2*N)])
    xc *= unbiased

    xcmax = np.max(xc[N - 300 : N+300])
    print "xcmax", xcmax, f1, f2
    return xcmax

#    np.savetxt("xc.txt", xc)


infiles = sys.argv[1:]

wiggles = [(np.loadtxt(f), f) for f in infiles]
print "loaded", len(wiggles), "wiggle files"
fpairs = itertools.combinations(wiggles, 2)

evf= "wiggles/6/66/2/4686108_BHZ_raw.dat"
p = [(xcorr(a, b, f1, f2), f1, f2) for ((a, f1), (b, f2)) in fpairs if f1 == evf or f2==evf]
p.sort()

print "lowest:"
for pair in p[:40]:
    print pair

print "highest:"
for pair in p[40:]:
    print pair

print len(p)
