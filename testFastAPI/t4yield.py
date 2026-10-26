def a():
    for i in range(5):
        yield i


a = a()
print(next(a))
print(next(a))
print(next(a))
