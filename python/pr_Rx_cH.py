import numpy as np

def f_I(x0):
    return np.abs(1 - x0)/2

def f_H(x0):
    # Calculate the complex part using sqrt(1 - x0)
    complex_part = np.sqrt(1 - x0) * 1j  # j is the imaginary unit in Python
    return np.abs((np.sqrt(1 + x0) + complex_part) / 2)

def f_3(x0, x1):
    # Calculate p1 using the formula provided
    p1 = (1 - x1) / 2
    return p1 * f_I(x0) + (1 - p1) * f_H(x0)

# Test the function with some example values
x0_test = np.linspace(-1, 1, 5)
x1_test = np.linspace(-1, 1, 5)
print("f3(x0, x1) for some example values:")
for x0 in x0_test:
    for x1 in x1_test:
        fI=f_I(x0)
        fH= f_H(x0)
        ft=f_3(x0, x1) 
        print("x0,x1=(%.2f,%.2f) --> fI,fH, f3 --> %.2f  %.2f  %.2f"%(x0,x1,fI,fH,ft))



'''  Latex formulas
f_3(x_0, x_1) = p_1 \cdot f_I(x_0) + (1 - p_1) \cdot f_H(x_0)

p_1 = \frac{1 - x_1}{2}

f_I(x_0) = \frac{|1 - x_0|}{2}

f_H(x_0) = \left| \frac{\sqrt{1 + x_0} + j \sqrt{1 - x_0}}{2} \right|



'''
