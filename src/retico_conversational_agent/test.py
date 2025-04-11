def f(a):
    def g():
        print(a)

    return g


print(f)
print(f(1))
print(f(1)())
print(f(1).__dir__())
print(f(1).__closure__[0])
print(f(1).__closure__[0].cell_contents)
