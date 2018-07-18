#Python is an interpreted programming language which means you can execute code line by line directly and freely, without previously compiling a program into machine-language instructions(like you do in C++/FORTRAN/C etc.)

# If a line starts with "#", it is a commented line and anything code(instruction) written in that line will not be executed


#First we import libraries which have the pre-written and well-defined functions

import numpy as np  
#numpy is the most useful library with support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.

import matplotlib.pyplot as plt
#matplotlib is a widely used plotting library and pyplot is a matplotlib module which provides a MATLAB-like interface.


#create an array x with values 1,2,3..,10
x=np.arange(1.0,11.0)

#check the values by printing to screen
print "The values in x are: ", x

#create an array y with values 10,9,8..,1
y=np.arange(10.0,0.0,-1)

#check the values by printing to screen
print "The values in y are: ", y


#check the shape of arrays
print "the shape of array x is: ", x.shape
print "the shape of array y is: ", y.shape

#elementwise operation on arrays
print "addition is: ", x+y
print "multiplication is: ", x*y


#elements of array
print "the first element of array x is: ", x[0]
print "the first element of array y is: ", y[0]

print "the last element of array x is: ", x[-1]
print "the last element of array y is: ", y[-1]


#for loop
for i in range(0,x.shape[0]):
        print "element number ",i," of array x is ", x[i]
        
 

#elementwise array operations
print "subtraction of first element of x with last element of y: ", x[0]-y[-1]
print "division of 3rd element of x by 5th element of y: ", x[2]/y[4]
print "(2nd element of x + 8th element of y) * (average of x) ", (x[1]+y[7])*(np.average(x))

#plotting

#simple plot
plt.plot(x,y)
plt.show()

#plot points
plt.plot(x,y,'o')
plt.show()

#plot points with line
plt.plot(x,y,'-o')
plt.show()

#plot red colored points
plt.plot(x,y,'ro')
plt.show()

#multiple plots with multiple colors and markers
plt.plot(x,y,'-go') #plots x vs y with a green colored line with green colored circles for data points
plt.plot(x,y+2.0,'--r') #plots x vs y+2.0(i.e. 2 added to each value of y array) with a dashed red colored line
plt.show()


#labeling plots
plt.plot(x,y,'-yo',label='y=y') #plots x vs y with a yellow colored line with yellow circles for data points
plt.plot(x,y*x,'-ks',label='y= y*x') #plots x vs y*x with a black colored line with black squares for data points
plt.plot(x,y+2.0,'--r',label='y= y+2.0') #plots x vs y+2.0(i.e. 2 added to each value of y array) with a dashed red colored line
plt.legend()
plt.xlabel('x values')
plt.ylabel('y values')
plt.title('Plot of x values with operations on y values')
plt.show()

