import diffsteps
from diffsteps import print_html_steps
from sympy import symbols, var
from sympy.parsing.latex import parse_latex

x = symbols('x')
#expr = parse_latex(r" 6x-\frac{3}{9x-6} ")
expr = parse_latex(r"x\tan{\sqrt{x}}")
print(expr)
solucion = print_html_steps(expr, x)
print(solucion)
