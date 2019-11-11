from sympy import *
from sympy.parsing.latex import parse_latex

expresion = parse_latex(r"\frac{2}{\sqrt{9x-17}}")
print(srepr(expresion))
print(expresion)
print(latex(diff(expresion)))
