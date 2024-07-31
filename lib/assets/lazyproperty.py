class LazyProperty:
    def __init__(self, getter, setter=None, deleter=None):
        """
        Creates a property that waits for the first use to be initialized. After this, it always returns the same
        result.

        Usage:

        class Bar:
            @LazyProperty
            def foo(self):
                print('calculating... ', end='')
                value = 1+1
                return value

            @foo.setter
            def foo(self, value):
                print('Setting foo with value automatically.')

        test = Bar()
        print(test.foo)  # print 'calculating... 2'
        print(test.foo)  # print '2'
        test.foo = 4     # print 'Setting foo with value automatically.'
        print(test.foo)  # print '4'. No calculate anymore.

        The value is stored in Bar._foo only once.

        :param getter:
        :param setter:
        """
        self.getter = getter
        self.setter = setter
        self.deleter = deleter
        self.name = self.getter.__name__
        self.attr_name = '_' + self.name

    def __get__(self, instance, owner):
        if instance is None:
            raise AttributeError(f"The descriptor must be used with class instances.")

        try:
            value = getattr(instance, self.attr_name)
            if value is None:
                raise AttributeError
        except AttributeError:
            value = self.getter(instance)
            setattr(instance, self.attr_name, value)

        return value

    def __set__(self, instance, value):
        if instance is None:
            raise AttributeError("The descriptor must be used with class instances.")

        if self.setter is None:
            raise AttributeError(f"Setter not defined for '{self.name}'")

        setattr(instance, self.attr_name, value)
        self.setter(instance, value)

    def __delete__(self, instance):
        if instance is None:
            raise AttributeError("The descriptor must be used with class instances.")

        if self.deleter is not None:
            self.deleter(instance)
        try:
            delattr(instance, '_' + self.name)
        except AttributeError:
            pass
