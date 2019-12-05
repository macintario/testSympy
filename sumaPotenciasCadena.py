from sympy import *
from sympy.parsing.latex import parse_latex


#Solución a derivada de y=u^n+w^m



salida = open("/tmp/salida.txt","w")
init_printing()
x = var('x')
u = var('u')
n = var('n')
v = var('v')
m = var('m')
u = parse_latex(r"5x-7")
n = 5
v = parse_latex(r"4-\frac{1}{6x^4}")
m = -1
print(u)
print(v)
f1=u**n
print(diff(f1))


