import diffsteps
from diffsteps import print_html_steps
from sympy import symbols, var
from sympy.parsing.latex import parse_latex

x = var('x')
#expr = parse_latex(r" 6x-\frac{3}{9x-6} ")
expr = parse_latex(r"\tan(\sqrt{x})")
solucion=print_html_steps(expr, x)
print(solucion)
