from a import helper, Foo


def use():
    f = Foo()
    return f.method() + helper(2)
