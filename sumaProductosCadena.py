from sympy import *
from sympy.parsing.latex import parse_latex


salida = open("/tmp/salida.txt","w")
init_printing()
x = var('x')
u = var('u')
f = parse_latex(r"\sqrt{u} ")
g = parse_latex(r" \frac{3}{5}-\frac{2}{7}x")
salida.write("$$f(x)=%s$$<br/>\n" % latex(f.subs(u,g)))
salida.write("$$f(u)=%s$$<br/>\n" % latex(f))
salida.write("$$g(x)=%s$$<br/>\n" % latex(g))
df = diff(f)
dg = diff(g)
salida.write("$$\\frac{df}{du}=%s$$<br/>\n" % latex(df))
salida.write("$$g(x)=%s$$<br/>\n" %latex(dg))
result = df*dg
salida.write("$$\\frac{df}{du}\\frac{du}{dx}=%s$$<br/>\n" % latex(result))
result = result.subs(u,g)
salida.write("$$\\frac{df}{dx}=%s=%s$$<br/>\n" % (latex(result), latex(result.cancel())))
#salida.write(sp.latex(result))
salida.close()