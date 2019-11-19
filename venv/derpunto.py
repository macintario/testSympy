from sympy import *
from sympy.parsing.latex import parse_latex

salida = open("/tmp/salida.txt","w")
init_printing()
x = var('x')
h = var('h')
f = parse_latex(r"\frac{2}{\sqrt{9x-17}}")
fh=f.subs(x,x+h)
drv=(fh-f)/h
salida.write("$$\\displaystyle f(x)=%s$$\n" %(latex(f)))
salida.write("$$\\displaystyle f(x+h)=%s$$\n" %latex(fh))
salida.write("$$\\displaystyle \\frac{f(x+h)-f(x)}{h}=%s$$\n" % latex(drv))
drv=limit(drv,h,0)
salida.write("$$\\displaystyle\\lim_{h\\to 0} \\frac{f(x+h)-f(x)}{h}=%s=%s$$\n" % (latex(drv),drv.doit()))
