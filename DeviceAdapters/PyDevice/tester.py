from test_devices import RandomGenerator, Camera


r = RandomGenerator()
devices = {'cam': Camera(random_generator=r), 'rng': r}
