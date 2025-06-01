from pyeda.inter import *

e1 = exprvar("e1")
e2 = exprvar("e2")
e3 = exprvar("e3")
e4 = exprvar("e4")
e5 = exprvar("e5")
e6 = exprvar("e6")
f1 = e1 ^ e5 ^ e6 ^ (e1 & e4) ^ (e2 & e3) ^ (e2 & e5) ^ (e4 & e5) ^ (e5 & e6)
f1 ^= (
    (e1 & e3 & e4)
    ^ (e1 & e3 & e6)
    ^ (e1 & e4 & e5)
    ^ (e2 & e3 & e6)
    ^ (e2 & e4 & e6)
    ^ (e3 & e5 & e6)
)
f1 ^= (
    (e1 & e2 & e3 & e4)
    ^ (e1 & e2 & e3 & e5)
    ^ (e1 & e2 & e5 & e6)
    ^ (e2 & e3 & e4 & e6)
    ^ (e1 & e2 & e3 & e4 & e5)
)
f1 ^= e1 & e3 & e4 & e5 & e6

print("f1 cnf form:")
print(f1.to_cnf())

print("espresso:")
(f1m,) = espresso_exprs(f1.to_dnf())
print("finished")
## TO cnf took around 270GB of Ram
print(f1m)
print("-----")
print("CNF: ")
##print(f1m.to_cnf())


"""
Returns: 
And(
    Or(e1, e2, ~e3, e4, ~e5, ~e6),
    Or(e1, ~e2, e3, ~e4, ~e5, ~e6),
    Or(e1, ~e2, e3, ~e4, e5, ~e6),
    Or(~e1, ~e2, e3, ~e4, e5, ~e6),
    Or(~e1, ~e2, e3, ~e4, ~e5, e6),
    Or(e1, e2, e3, ~e4, ~e5, ~e6),
    Or(e1, e2, e5, e6),
    Or(e1, e2, ~e4, ~e5, e6),
    Or(e1, e3, e5, e6),
    Or(e1, ~e2, e3, e4, ~e5, e6),
    Or(e1, ~e2, ~e3, ~e4, ~e5, e6),
    Or(~e1, e3, e4, e5, ~e6),
    Or(e1, ~e2, e3, e4, ~e5, ~e6),
    Or(~e1, e2, e3, e4, ~e5, ~e6),
    Or(~e1, e3, ~e4, e5, e6),
    Or(~e1, ~e2, ~e3, e4, e5, e6),
    Or(~e1, e2, e4, ~e5, e6),
    Or(~e1, e2, ~e3, ~e4, ~e5, e6),
    Or(~e1, ~e2, ~e3, ~e4, e5, ~e6),
    Or(~e1, e2, ~e3, e4, ~e5, ~e6),
    Or(e1, ~e2, ~e3, ~e4, ~e5, ~e6),
    Or(~e1, ~e2, e3, e4, ~e5, ~e6),
    Or(~e1, ~e2, e3, ~e4, ~e5, ~e6),
    Or(~e1, ~e2, ~e3, ~e4, ~e5, ~e6),
)
"""
print("-----")

f_0 = exprvar("f0")
f_1 = exprvar("f1")
f_2 = exprvar("f2")
f_5 = exprvar("f5")

print("f bit roatate cnf")
f2 = f_0 ^ f_1 ^ f_2 ^ f_5
print(f2.to_cnf())
(f2m,) = espresso_exprs(f2.to_dnf())
print("f bit roatate espresso cnf")
print(f2m.to_cnf())

print("f bit roatate espresso dnf")
print(f2m)
print("-----")

t_8 = exprvar("t8")
u_28 = exprvar("u28")

t9 = t_8 ^ u_28
print(t9.to_cnf())


def convert_to_rust_clause(cnf):
    print(type(cnf))


convert_to_rust_clause(f2.to_cnf())
