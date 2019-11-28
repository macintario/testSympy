from sympy import *
from sympy.parsing.latex import parse_latex
salida = open("/tmp/solucion.txt","w")
init_printing()
x = var('x')
h = var('h')
f = parse_latex(r" 6x-\frac{3}{9x-6} ")
fh=f.subs(x,x+h)
drv=(fh-f)/h
salida.write("$$\\displaystyle f(x)=%s$$<br/><br/>\n" %(latex(f)))
salida.write("$$\\displaystyle f(x+h)=%s$$<br/><br/>\n" %latex(fh))
salida.write("$$\\displaystyle \\frac{f(x+h)-f(x)}{h}=%s=%s $$\n <br/><br/>" % (latex(drv),latex(drv.factor())))
drv=limit(drv,h,0)
resultado=drv.apart()
salida.write("$$\\displaystyle f'(x)=\\lim_{h\\to 0} \\frac{f(x+h)-f(x)}{h}=%s=%s$$<br/><br/>\n" % (latex(drv),latex(resultado)))
print(resultado)