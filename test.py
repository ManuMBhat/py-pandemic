class Test(object):
    def __init__(self,a):
        self.a = a


ob = Test(5)
print(id(ob))
ob.a = 123456789
print(id(ob))