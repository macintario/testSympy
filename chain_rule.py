from sympy import *
from sympy.parsing.latex import parse_latex


salida = open("/tmp/salida.txt","w")
init_printing()
x = var('x')
u = var('u')
f = parse_latex(r"\sqrt{u} ")
g = parse_latex(r" \frac{3}{5}-\frac{2}{7}x")
#print(latex(f.subs))
df = diff(f)
dg = diff(g)
print(latex(df))
print(latex(dg))
result = df*dg
print(latex(result))
result = result.subs(u,g)
print(latex(result))
#salida.write(sp.latex(result))
salida.close()