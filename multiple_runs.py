# script to run the function several times and changing the input parameters

import tumorlib
import numpy

I = 10
J= 10

b = numpy.zeros((I,J))

for i in range(I):
    for j in range(J):
        a = tumorlib.tumorfunc(30, 10, i, j)
        print('accuracy for',i,',',j,'is',a)
        b[i,j]=a


print('mean accuracy:',numpy.mean(b))